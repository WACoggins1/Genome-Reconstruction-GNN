
#!/usr/bin/env python3
"""
Build a unitig‑compressed de Bruijn graph from a FASTA.

Pipeline:
1. Parse FASTA (each record = chromosome)
2. k‑mer segmentation (user‑specified k)
3. Build de Bruijn graph
4. Label true edges
5. Compress into unitigs
6. Save graph_data.pkl with:
     - node_features (unitig features)
     - edges (unitig‑level edges)
     - edge_labels (unitig‑level labels)
     - node_chroms (per‑unitig chromosome)
"""

import argparse
import pickle
from collections import defaultdict
from Bio import SeqIO
import numpy as np

# one‑hot for A,C,G,T
nucleotide_to_onehot = {
    'A': [1,0,0,0],
    'C': [0,1,0,0],
    'G': [0,0,1,0],
    'T': [0,0,0,1]
}

def onehot_encode_kmer(kmer):
    """Flattened one‑hot for a k‑mer."""
    vec = []
    for nuc in kmer:
        vec.extend(nucleotide_to_onehot.get(nuc, [0,0,0,0]))
    return vec

def process_chromosome(chrom_id, seq, k):
    """Return list of node dicts and ground‑truth order for one chromosome."""
    nodes, order = [], []
    n = len(seq)
    for pos in range(n - k + 1):
        kmer = seq[pos:pos+k]
        node = {'id':f"{chrom_id}_{pos}", 'chrom':chrom_id, 'pos':pos, 'kmer':kmer}
        nodes.append(node)
        order.append(node['id'])
    return nodes, order

def build_graph(nodes, overlap):
    """Build raw k‑mer overlap edges (labelled later)."""
    prefix = defaultdict(list)
    for node in nodes:
        prefix[node['kmer'][:overlap]].append(node['id'])
    edge_dict = {}
    for node in nodes:
        suf = node['kmer'][-overlap:]
        for tgt in prefix.get(suf, []):
            edge_dict[(node['id'],tgt)] = 0
    return edge_dict

def label_groundtruth(edge_dict, gt_order):
    """Mark consecutive k‑mers along each chromosome as true edges."""
    for a,b in zip(gt_order, gt_order[1:]):
        edge_dict[(a,b)] = 1
    return edge_dict

def compress_unitigs(edges):
    """
    Collapse maximal non‑branching chains into unitigs.
    Returns comp_id (orig_node_idx→unitig_id) and list of unitig edges.
    """
    in_nb, out_nb = defaultdict(list), defaultdict(list)
    for u,v in edges:
        out_nb[u].append(v)
        in_nb[v].append(u)

    comp_id, next_id = {}, 0
    for node in set(in_nb)|set(out_nb):
        if node in comp_id: continue
        u = node
        # walk back to chain start
        while len(in_nb[u])==1 and len(out_nb[in_nb[u][0]])==1:
            u = in_nb[u][0]
        # walk forward and collapse
        comp_id[u] = next_id
        v = u
        while len(out_nb[v])==1 and len(in_nb[out_nb[v][0]])==1:
            v = out_nb[v][0]
            comp_id[v] = next_id
        next_id += 1

    # build unitig edge set
    unitig_edges = set()
    for u,v in edges:
        cu,cv = comp_id[u], comp_id[v]
        if cu!=cv:
            unitig_edges.add((cu,cv))

    # return a list mapping 0..N-1 orig_node→unitig and the edges
    return [comp_id[i] for i in range(len(comp_id))], list(unitig_edges)

def main(input_fasta, output_pickle, k):
    overlap = k - 1

    # 1) Parse & build raw graph
    all_nodes, all_gt = [], []
    for rec in SeqIO.parse(input_fasta, "fasta"):
        seq = str(rec.seq).upper()
        nodes, order = process_chromosome(rec.id, seq, k)
        all_nodes.extend(nodes)
        all_gt.extend(order)
        print(f"Parsed {rec.id}: {len(nodes)} k‑mers")

    edge_dict = build_graph(all_nodes, overlap)
    edge_dict = label_groundtruth(edge_dict, all_gt)
    print(f"Raw edges: {len(edge_dict)}")

    # 2) Node features + node_chroms
    idx_map = {node['id']:i for i,node in enumerate(all_nodes)}
    feature_matrix = np.array([onehot_encode_kmer(n['kmer']) for n in all_nodes],dtype=np.float32)
    node_chroms    = [n['chrom'] for n in all_nodes]

    # 3) Raw edge list + labels
    edges, labels = [], []
    for (s,t),lab in edge_dict.items():
        if s in idx_map and t in idx_map:
            edges.append((idx_map[s], idx_map[t]))
            labels.append(lab)

    # 4) Unitig compression
    comp_id, unitig_edges = compress_unitigs(edges)
    # unitig features (average of members)
    members = defaultdict(list)
    for orig,uid in enumerate(comp_id):
        members[uid].append(orig)
    unitig_feats = np.stack([feature_matrix[members[u]].mean(0) for u in sorted(members)])
    # unitig labels (OR over underlying edges)
    ulabel = {}
    for (u,v),lab in zip(edges,labels):
        cu,cv = comp_id[u], comp_id[v]
        if cu!=cv:
            ulabel[(cu,cv)] = ulabel.get((cu,cv),0) or lab
    unitig_labels = [ulabel[e] for e in sorted(ulabel)]
    unitig_edge_list = sorted(ulabel)

    # unitig chroms
    unitig_chroms = []
    for u in range(len(members)):
        orig0 = members[u][0]
        unitig_chroms.append(node_chroms[orig0])

    # 5) Save compressed graph
    graph_data = {
        'node_features': unitig_feats,      # [num_unitigs, feat_dim]
        'edges':         unitig_edge_list,  # [(cu,cv),...]
        'edge_labels':   unitig_labels,     # [0/1,...]
        'node_chroms':   unitig_chroms      # [chrom,...]
    }
    with open(output_pickle, 'wb') as f:
        pickle.dump(graph_data, f)
    print(f"Wrote compressed graph with {len(unitig_feats)} unitigs to {output_pickle}")

if __name__=="__main__":
    p = argparse.ArgumentParser(description="Unitig‑compressed de Bruijn graph builder")
    p.add_argument("input_fasta")
    p.add_argument("output_pickle")
    p.add_argument("--k", type=int, default=31)
    args = p.parse_args()
    main(args.input_fasta, args.output_pickle, args.k)
