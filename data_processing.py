#!/usr/bin/env python3
"""
Build raw and unitig-compressed de Bruijn graphs from a FASTA.

Pipeline:
1. Parse FASTA (each record = chromosome)
2. k‑mer segmentation (user‑specified k)
3. Build de Bruijn graph
4. Label true edges
5. Save raw graph
6. Compress into unitigs
7. Save unitig-compressed graph
"""
import argparse
import pickle
from collections import defaultdict
from Bio import SeqIO
import numpy as np

# one-hot for A,C,G,T
nucleotide_to_onehot = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1]
}

def onehot_encode_kmer(kmer):
    """Flattened one-hot for a k-mer."""
    vec = []
    for nuc in kmer:
        vec.extend(nucleotide_to_onehot.get(nuc, [0, 0, 0, 0]))
    return vec


def process_chromosome(chrom_id, seq, k):
    """Return list of node dicts and ground-truth order for one chromosome."""
    nodes, order = [], []
    n = len(seq)
    for pos in range(n - k + 1):
        kmer = seq[pos:pos + k]
        node = {'id': f"{chrom_id}_{pos}", 'chrom': chrom_id, 'pos': pos, 'kmer': kmer}
        nodes.append(node)
        order.append(node['id'])
    return nodes, order


def build_graph(nodes, overlap):
    """Build raw k-mer overlap edges (labelled later)."""
    prefix = defaultdict(list)
    for node in nodes:
        prefix[node['kmer'][:overlap]].append(node['id'])
    edge_dict = {}
    for node in nodes:
        suf = node['kmer'][-overlap:]
        for tgt in prefix.get(suf, []):
            edge_dict[(node['id'], tgt)] = 0
    return edge_dict


def label_groundtruth(edge_dict, gt_order):
    """Mark consecutive k-mers along each chromosome as true edges."""
    for a, b in zip(gt_order, gt_order[1:]):
        edge_dict[(a, b)] = 1
    return edge_dict


def compress_unitigs(edges):
    """
    Collapse maximal non-branching chains into unitigs.
    Returns comp_id (orig_node_idx→unitig_id) and list of unitig edges.
    """
    in_nb, out_nb = defaultdict(list), defaultdict(list)
    for u, v in edges:
        out_nb[u].append(v)
        in_nb[v].append(u)

    comp_id, next_id = {}, 0
    for node in set(in_nb) | set(out_nb):
        if node in comp_id:
            continue
        u = node
        # walk back to chain start
        while len(in_nb[u]) == 1 and len(out_nb[in_nb[u][0]]) == 1:
            u = in_nb[u][0]
        # walk forward and assign unitig
        comp_id[u] = next_id
        v = u
        while len(out_nb[v]) == 1 and len(in_nb[out_nb[v][0]]) == 1:
            v = out_nb[v][0]
            comp_id[v] = next_id
        next_id += 1

    # build unitig edge set
    unitig_edges = set()
    for u, v in edges:
        cu, cv = comp_id[u], comp_id[v]
        if cu != cv:
            unitig_edges.add((cu, cv))

    return [comp_id[i] for i in range(len(comp_id))], list(unitig_edges)


def main(input_fasta, k, prefix):
    overlap = k - 1

    # 1) Parse & build raw graph
    all_nodes, all_gt = [], []
    for rec in SeqIO.parse(input_fasta, "fasta"):
        seq = str(rec.seq).upper()
        nodes, order = process_chromosome(rec.id, seq, k)
        all_nodes.extend(nodes)
        all_gt.extend(order)
        print(f"Parsed {rec.id}: {len(nodes)} k-mers")

    edge_dict = build_graph(all_nodes, overlap)
    edge_dict = label_groundtruth(edge_dict, all_gt)
    print(f"Raw edges: {len(edge_dict)}")

    # 2) Node features + node_chroms
    idx_map = {node['id']: i for i, node in enumerate(all_nodes)}
    feature_matrix = np.array(
        [onehot_encode_kmer(n['kmer']) for n in all_nodes],
        dtype=np.float32
    )
    node_chroms = [n['chrom'] for n in all_nodes]

    # 3) Raw edge list + labels
    edges, labels = [], []
    for (s, t), lab in edge_dict.items():
        if s in idx_map and t in idx_map:
            edges.append((idx_map[s], idx_map[t]))
            labels.append(lab)

    # 4) Save raw graph
    raw_graph = {
        'node_features': feature_matrix,
        'edges': edges,
        'edge_labels': labels,
        'node_chroms': node_chroms
    }
    raw_path = f"{prefix}_raw.pkl"
    with open(raw_path, 'wb') as f:
        pickle.dump(raw_graph, f)
    print(f"Wrote raw graph with {feature_matrix.shape[0]} nodes to {raw_path}")

    # 5) Unitig compression
    comp_id, unitig_edges = compress_unitigs(edges)
    members = defaultdict(list)
    for orig, uid in enumerate(comp_id):
        members[uid].append(orig)
    # unitig features (average of member k-mers)
    unitig_feats = np.stack([
        feature_matrix[members[u]].mean(0) for u in sorted(members)
    ])
    # unitig labels (OR over underlying edges)
    ulabel = {}
    for (u, v), lab in zip(edges, labels):
        cu, cv = comp_id[u], comp_id[v]
        if cu != cv:
            ulabel[(cu, cv)] = ulabel.get((cu, cv), 0) or lab
    unitig_labels = [ulabel[e] for e in sorted(ulabel)]
    unitig_edge_list = sorted(ulabel)
    # unitig chroms
    unitig_chroms = []
    for u in range(len(members)):
        orig0 = members[u][0]
        unitig_chroms.append(node_chroms[orig0])

    unitig_graph = {
        'node_features': unitig_feats,
        'edges': unitig_edge_list,
        'edge_labels': unitig_labels,
        'node_chroms': unitig_chroms
    }
    unitig_path = f"{prefix}_unitig.pkl"
    with open(unitig_path, 'wb') as f:
        pickle.dump(unitig_graph, f)
    print(f"Wrote unitig graph with {unitig_feats.shape[0]} unitigs to {unitig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build raw and unitig-compressed de Bruijn graphs"
    )
    parser.add_argument("input_fasta", help="Input FASTA file path")
    parser.add_argument(
        "--k", type=int, default=31,
        help="k-mer size (default: 31)"
    )
    parser.add_argument(
        "--prefix", default="graph_data",
        help="Output file prefix (default: graph_data)"
    )
    args = parser.parse_args()
    main(args.input_fasta, args.k, args.prefix)
