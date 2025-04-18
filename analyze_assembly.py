#!/usr/bin/env python3
"""
analyze_assembly.py

– Contig = a contiguous reference sequence (e.g. a chromosome or scaffold in your FASTA).
– Unitig = a maximal non‑branching chain in the de Bruijn graph collapsed into one node.

This script:
 1) Loads the unitig‑compressed graph + GNN checkpoint
 2) Tunes a threshold on validation edges for best F1
 3) Reconstructs the Hamiltonian cycle with a branch‑aware hybrid:
     • Greedy until greedy_frac covered
     • Beam only at true forks
 4) Reports global & per‑chromosome metrics
"""

import argparse
import pickle
import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch_geometric.data import Data
from train_model import EdgeClassifier, create_data_object, split_edge_data

# ----------------------------------------
# Branch‑aware hybrid reconstruction

def branch_aware_cycle(data, model,
                       greedy_frac=0.3,
                       beam_width=5,
                       top_k=10,
                       log_every=200_000):
    """
    1) Greedy walk until greedy_frac of nodes visited.
    2) Branch‑aware beam for the rest:
       – If node has only 1 outgoing, extend greedily.
       – Else do dynamic top_k + fallback beam expansions.
    Guard against nodes with no outgoing edges.
    """
    model.eval()
    # precompute scores
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        scores = torch.sigmoid(logits).cpu().numpy()
    # build adjacency
    srcs, tgts = data.edge_index.cpu().numpy()
    out_edges = defaultdict(list)
    for s, t, sc in zip(srcs, tgts, scores):
        out_edges[s].append((t, sc))
    # detect branch points
    branch_nodes = {u for u, neis in out_edges.items() if len(neis) > 1}

    N = data.num_nodes
    cutoff = int(N * greedy_frac)

    # --- Greedy phase ---
    visited = {0}
    cycle = [0]
    while len(visited) < cutoff:
        curr = cycle[-1]
        neigh = out_edges.get(curr, [])
        if not neigh:
            print(f"[Greedy] no outgoing edges at node {curr}, stopping greedy.")
            break
        # local dynamic top_k
        topk = sorted(neigh, key=lambda x: -x[1])[:top_k]
        # pick first unvisited
        for nxt, sc in topk:
            if nxt not in visited:
                cycle.append(nxt)
                visited.add(nxt)
                break
        else:
            print(f"[Greedy] no unvisited in top_{top_k} at node {curr}, stopping.")
            break
        if len(cycle) % log_every == 0:
            print(f"[Greedy] {len(visited)}/{N}")

    print(f"[Hybrid] switch to beam at {len(visited)}/{N}")

    # --- Beam phase ---
    beam = [(cycle, visited, 0.0)]
    final, best_rep = [], len(visited)
    iteration = 0

    while beam:
        iteration += 1
        if iteration % 1000 == 0:
            print(f"[Beam] iter {iteration}, beam_size={len(beam)}")

        candidates = []
        for path, vs, lp in beam:
            curr = path[-1]
            neigh = out_edges.get(curr, [])
            if not neigh:
                continue

            # no branch → greedy extension
            if curr not in branch_nodes:
                nxt, sc = max(neigh, key=lambda x: x[1])
                if nxt not in vs:
                    candidates.append((path + [nxt], vs | {nxt}, lp + np.log(sc + 1e-9)))
                continue

            # branch → dynamic top_k + fallback
            topk = sorted(neigh, key=lambda x: -x[1])[:top_k]
            unvis = [(n, sc) for n, sc in topk if n not in vs]
            if not unvis:
                for n, sc in sorted(neigh, key=lambda x: -x[1]):
                    if n not in vs:
                        unvis = [(n, sc)]
                        break
            for nxt, sc in unvis:
                candidates.append((path + [nxt], vs | {nxt}, lp + np.log(sc + 1e-9)))

        if not candidates:
            print("[Beam] no expansions → stopping")
            break

        # keep best B
        candidates.sort(key=lambda x: -x[2])
        beam = candidates[:beam_width]

        # check for full cycles
        for path, vs, lp in beam:
            if len(vs) == N:
                final.append((path, lp))

        # log coverage
        topcov = len(beam[0][1])
        if topcov >= best_rep + log_every:
            best_rep = topcov
            print(f"[Beam] coverage {topcov}/{N}")

        if final:
            break

    return max(final, key=lambda x: x[1])[0] if final else max(beam, key=lambda x: len(x[1]))[0]

# ----------------------------------------
# Evaluation harness

def evaluate_cycle(model, data, full_labels, threshold):
    n = data.num_nodes
    cycle = branch_aware_cycle(data, model)
    cycle_edges = {(cycle[i], cycle[i+1]) for i in range(len(cycle)-1)}
    gt_edges    = {(i, i+1) for i in range(n-1)}

    tp = len(cycle_edges & gt_edges)
    fp = len(cycle_edges - gt_edges)
    fn = len(gt_edges - cycle_edges)
    prec = tp/(tp+fp) if tp+fp else 0.0
    rec  = tp/(tp+fn) if tp+fn else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    cov  = len(set(cycle))/n

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        scores = torch.sigmoid(logits).cpu().numpy()
    preds = (scores >= threshold).astype(int)
    cm    = confusion_matrix(full_labels, preds)

    return {'prec':prec, 'rec':rec, 'f1':f1, 'cov':cov, 'cm':cm}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pickle_file")
    p.add_argument("checkpoint")
    p.add_argument("--hidden", type=int, default=32)
    args = p.parse_args()

    gd = pickle.load(open(args.pickle_file,'rb'))
    full_labels = gd['edge_labels']
    node_chroms = gd['node_chroms']

    data   = create_data_object(gd)
    splits = split_edge_data(gd)

    device = data.x.device
    model  = EdgeClassifier(data.num_node_features, args.hidden).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # tune threshold
    with torch.no_grad():
        val_logits = model(data.x, data.edge_index)[splits['val_idx'].to(device)]
        val_scores = torch.sigmoid(val_logits).cpu().numpy()
        val_truths = splits['val_labels'].cpu().numpy()
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.linspace(0.1, 0.9, 81):
        f = f1_score(val_truths, val_scores >= thr, zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, thr
    print(f"Threshold={best_thr:.3f} → Val F1={best_f1:.3f}\n")

    # global
    gm = evaluate_cycle(model, data, full_labels, best_thr)
    print("=== GLOBAL ===")
    print(f"Prec={gm['prec']:.3f} Rec={gm['rec']:.3f} F1={gm['f1']:.3f} Cov={gm['cov']:.3f}")
    print("Confusion:", gm['cm'], "\n")

    print("=== PER‑CHR ===")
    chrom_to_idxs = defaultdict(list)
    for i,ch in enumerate(node_chroms):
        chrom_to_idxs[ch].append(i)

    for ch,idxs in chrom_to_idxs.items():
        mask = np.isin(data.edge_index[0].cpu(), idxs) & np.isin(data.edge_index[1].cpu(), idxs)
        sub_ei = data.edge_index[:, mask]
        full_labels_arr = np.array(full_labels)
        # directly index numpy array with boolean mask
        sub_labels = full_labels_arr[mask]

        old2new = {o:i for i,o in enumerate(idxs)}
        srcs, tgts = sub_ei.cpu().numpy()
        sub_ei = torch.tensor([[old2new[x] for x in srcs], [old2new[x] for x in tgts]], device=device)
        sub_x = data.x[idxs]
        sub_data = Data(x=sub_x, edge_index=sub_ei)

        m = evaluate_cycle(model, sub_data, sub_labels, best_thr)
        print(f"{ch:8s} Prec={m['prec']:.3f} Rec={m['rec']:.3f} F1={m['f1']:.3f} Cov={m['cov']:.3f}")

if __name__ == "__main__":
    main()
