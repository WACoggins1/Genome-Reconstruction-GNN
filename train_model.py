#!/usr/bin/env python3
"""
train_model.py

Pipeline to:
1. Load graph data (node features, edges, edge labels).
2. Split edges into train/validation/test with stratification.
3. Create a PyTorch Geometric Data object and move it to device.
4. Define an edge classification GNN (two-layer GCN + MLP).
5. Train the model, tracking loss, accuracy, precision, recall, and F1; save checkpoints.
6. Reconstruct the Hamiltonian cycle via greedy search on predicted edge scores.

Usage:
    python train_model.py <graph_data_pickle> [--epochs E] [--lr LR] [--wd WD] [--patience P]

Example:
    python train_model.py graph_data.pkl --epochs 100 --lr 0.01 --wd 1e-4 --patience 10
"""

import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
from concurrent.futures import ThreadPoolExecutor as ProcessPoolExecutor

def _hybrid_expand_one(path, visited_set, lp, out_edges, top_k):
    """Top‑level helper so it can be pickled by ProcessPoolExecutor."""
    curr = path[-1]
    neigh = sorted(out_edges.get(curr, []), key=lambda x: -x[1])[:top_k]
    results = []
    for nxt, sc in neigh:
        if nxt in visited_set:
            continue
        new_path    = path + [nxt]
        new_visited = visited_set | {nxt}
        new_lp      = lp + np.log(sc + 1e-9)
        results.append((new_path, new_visited, new_lp))
    return results

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)





class EdgeClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        src, tgt = edge_index
        edge_feat = torch.cat([x[src], x[tgt]], dim=1)
        return self.edge_mlp(edge_feat).squeeze()

def load_graph_data(pickle_file):
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)

def create_data_object(graph_data):
    x = torch.tensor(graph_data['node_features'], dtype=torch.float).to(device)
    edge_index = torch.tensor(graph_data['edges'], dtype=torch.long).t().contiguous().to(device)
    return Data(x=x, edge_index=edge_index)

def split_edge_data(graph_data, test_size=0.15, val_size=0.15):
    edges = np.array(graph_data['edges'])
    labels = np.array(graph_data['edge_labels'])
    indices = np.arange(len(labels))

    train_val_idx, test_idx, train_val_labels, test_labels = train_test_split(
        indices, labels, test_size=test_size, random_state=42, stratify=labels
    )
    val_fraction = val_size / (1 - test_size)
    train_idx, val_idx, train_labels, val_labels = train_test_split(
        train_val_idx, train_val_labels, test_size=val_fraction,
        random_state=42, stratify=train_val_labels
    )

    return {
        'train_idx': torch.tensor(train_idx, dtype=torch.long, device=device),
        'val_idx':   torch.tensor(val_idx,   dtype=torch.long, device=device),
        'test_idx':  torch.tensor(test_idx,  dtype=torch.long, device=device),
        'train_labels': torch.tensor(train_labels, dtype=torch.float, device=device),
        'val_labels':   torch.tensor(val_labels,   dtype=torch.float, device=device),
        'test_labels':  torch.tensor(test_labels,  dtype=torch.float, device=device),
    }

def evaluate(model, data, edge_idx, true_labels):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)[edge_idx]
        loss = F.binary_cross_entropy_with_logits(logits, true_labels)
        preds = (torch.sigmoid(logits) >= 0.5).cpu().numpy()
        truths = true_labels.cpu().numpy()
        return (
            loss.item(),
            accuracy_score(truths, preds),
            precision_score(truths, preds, zero_division=0),
            recall_score(truths, preds, zero_division=0),
            f1_score(truths, preds, zero_division=0)
        )

def train_model(model, data, splits, epochs, lr, wd, patience):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True) #scheduler to smooth out late-stage spikes in training
    best_val_loss = float('inf')
    patience_counter = 0

    metrics = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_prec': [], 'val_prec': [],
        'train_rec': [], 'val_rec': [],
        'train_f1': [], 'val_f1': []
    }

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy_with_logits(
            logits[splits['train_idx']],
            splits['train_labels']
        )
        loss.backward()
        optimizer.step()

        tr_loss, tr_acc, tr_prec, tr_rec, tr_f1 = evaluate(
            model, data, splits['train_idx'], splits['train_labels']
        )
        va_loss, va_acc, va_prec, va_rec, va_f1 = evaluate(
            model, data, splits['val_idx'], splits['val_labels']
        )

        metrics['train_loss'].append(tr_loss)
        metrics['val_loss'].append(va_loss)
        metrics['train_acc'].append(tr_acc)
        metrics['val_acc'].append(va_acc)
        metrics['train_prec'].append(tr_prec)
        metrics['val_prec'].append(va_prec)
        metrics['train_rec'].append(tr_rec)
        metrics['val_rec'].append(va_rec)
        metrics['train_f1'].append(tr_f1)
        metrics['val_f1'].append(va_f1)

        print(f"Epoch {epoch:03d} | Train Loss: {tr_loss:.4f}, Val Loss: {va_loss:.4f} | Train Acc: {tr_acc:.4f}, Val Acc: {va_acc:.4f}")
        scheduler.step(va_loss)

        if epoch %10 == 0:
            _, _, p, r, f1 = evaluate(model, data, splits['val_idx'], splits['val_labels'])
            print(f" -> Val P/R/F1 @epoch {epoch}: {p:.3f}/{r:.3f}/{f1:.3f}")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_checkpoint.pt")
            print("Checkpoint saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    epochs_range = range(1, len(metrics['train_loss']) + 1)
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, metrics['train_loss'], label='Train Loss')
    plt.plot(epochs_range, metrics['val_loss'],   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, metrics['train_acc'], label='Train Acc')
    plt.plot(epochs_range, metrics['val_acc'],   label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy')
    plt.tight_layout()
    plt.show()

    return model


def beam_cycle_dynamic_with_fallback(data, model,
                                     greedy_frac=0.5,
                                     beam_width=3,
                                     top_k=5,
                                     log_every=200_000):
    # Greedy search
    N = data.num_nodes
    threshold = int(N* greedy_frac)

    #Greedy greedy boy
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(data.x, data.edge_index)).cpu().numpy()
    scrs, tgts = data.edge_index.cpu().numpy()

    raw_edges = {}
    for s, t, sc in zip(scrs, tgts, scores):
        raw_edges.setdefault(s, []).append((t,sc))

    visited = {0}
    cycle = [0]

    while len(visited) < threshold:
        curr = cycle[-1]
        neigh = raw_edges.get(curr, [])
        # local dynamic top_k
        candidates = sorted(neigh, key=lambda x: -x[1])[:top_k]
        for nxt, sc in candidates:
            if nxt not in visited:
                cycle.append(nxt)
                visited.add(nxt)
                break
        else:
            break

        if len(cycle)% log_every == 0:
            print(f"[Greedy] visited {len(visited)}/{N}")

    print("Swapping to beam search...")


    # 2) Beam phase (dynamic prune + fallback)
    beam = [(cycle, visited, 0.0)]
    final = []
    best_reported = len(visited)
    iteration = 0

    while beam:
        iteration += 1
        if iteration % 1000 == 0:
            print(f"[Beam] iter={iteration}, beam_size={len(beam)}")

        candidates = []
        for path, vs, lp in beam:
            curr = path[-1]
            neigh = raw_edges.get(curr, [])  # full neighbor list

            # first try the top_k
            local = sorted(neigh, key=lambda x: -x[1])[:top_k]
            unvis = [(n, sc) for (n, sc) in local if n not in vs]

            # if that fails, fallback to the single best from full list
            if not unvis:
                for n, sc in sorted(neigh, key=lambda x: -x[1]):
                    if n not in vs:
                        unvis = [(n, sc)]
                        break

            # now expand along whatever we got
            for nxt, sc in unvis:
                candidates.append((
                    path + [nxt],
                    vs | {nxt},
                    lp + np.log(sc + 1e-9)
                ))

        if not candidates:
            print("[Beam] no more expansions (even after fallback)")
            break

        # keep the top B by cumulative log‑prob
        candidates.sort(key=lambda x: -x[2])
        beam = candidates[:beam_width]

        # check for completion
        for path, vs, lp in beam:
            if len(vs) == N:
                final.append((path, lp))

        # log coverage of best beam
        best_len = len(beam[0][1])
        if best_len >= best_reported + log_every:
            best_reported = best_len
            print(f"[Beam] best covers {best_len}/{N}")

        if final:
            break

    # return result
    return (max(final, key=lambda x: x[1])[0]
            if final else
            max(beam, key=lambda x: len(x[1]))[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pickle_file", help="Graph data pickle")
    parser.add_argument("--epochs",   type=int,   default=100,  help="Num epochs")
    parser.add_argument("--lr",       type=float, default=0.01, help="Learning rate")
    parser.add_argument("--wd",       type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int,   default=10,   help="Early stopping patience")
    args = parser.parse_args()

    graph_data = load_graph_data(args.pickle_file)
    data = create_data_object(graph_data)
    splits = split_edge_data(graph_data)
    model = EdgeClassifier(data.num_node_features, hidden_channels=32).to(device)

    model = train_model(
        model, data, splits,
        epochs=args.epochs,
        lr=args.lr,
        wd=args.wd,
        patience=args.patience
    )

    model.load_state_dict(torch.load("best_model_checkpoint.pt"))
    test_metrics = evaluate(model, data, splits['test_idx'], splits['test_labels'])
    print("Test  Loss: {:.4f}, Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}".format(*test_metrics))

    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, data, splits['test_idx'], splits['test_labels'])
    print(f"Test P/R/F1 {test_prec: .3f}/{test_rec:.3f}/{test_f1:.3f}")

    #confusion matrix
    logits = model(data.x, data.edge_index)[splits['test_idx']]
    preds = (torch.sigmoid(logits) >=0.5).cpu().numpy()
    truths = splits['test_labels'].cpu().numpy()
    print("Confusion Matrix: \n", confusion_matrix(truths, preds))


    #Don't print this out - it's going to eat the entire console. Save your eyes.
    #cycle = greedy_cycle_reconstruction(data, model)

    #print("Reconstructed cycle (node indices):", cycle)

if __name__ == "__main__":
    main()
