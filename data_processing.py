''' data processing script

Pipeline to:
1. Parse a FASTA file (each sequence assumed to be a chromosome)
2. Segment each chromosome into k-mers (k = 31).
3. Build the de Bruijn graph.
4. Label the edges.
5. Save the result.

Usage:
    python data_processing.py <input_fasta> <output_pickle>

Example:
    python data_processing.py human_genome.fasta graph_data.pkl
'''

import sys
import pickle
from Bio import SeqIO
import numpy as np

#Constants
K = 31
overlap = K - 1

#one-hot encoding for nucleotides
nucleotide_to_onehot = {
    'A': [1,0,0,0],
    'C': [0,1,0,0],
    'G': [0,0,1,0],
    'T': [0,0,0,1]
}

def onhot_encode_kmer(kmer):
    #Encode a k-mer into a flatened one-hot vector

    encoding = []
    for nuc in kmer.upper():
        encoding.extend(nucleotide_to_onehot.get(nuc, [0,0,0,0]))

    return encoding

def process_chromosome(chrom_id, sequence):
    '''
    For one chromosome, generate nodes and ground truth ordering.
    Each node is a dictionary with keys: 'id', 'chrom', 'pos', 'kmer'.
    We also record the ground truth ordering as a list of node ids in order
    '''

    nodes = []
    gt_order = [] #ground truth order
    n = len(sequence)
    node_id = 0

    #create a node for each k-mer
    for pos in range(n - K + 1):
        kmer = sequence[pos:pos+K]
        node = {
            'id':f"{chrom_id}_{node_id}", #create unique id using chromosome and local index
            'chrom': chrom_id,
            'pos': pos,
            'kmer': kmer
        }

        nodes.append(node)
        gt_order.append(node['id'])
        node_id += 1


    return nodes, gt_order

def build_graph(nodes):
    """
    Build a directed graph from nodes using de Bruijn overlap.
    For each node, compare last 30 bases to first 30 of next. If they match, connect them.
    """

    #Build dictionaries to quickly look up nodes by prefix
    prefix_dict= {}

    for node in nodes:
        prefix = node['kmer'][:overlap]
        prefix_dict.setdefault(prefix, []).append(node['id'])

    #create a mapping from node id to node for convenience
    id_to_node = {node['id']: node for node in nodes}

    #Build edges
    edge_dict = {}

    #First ad all edges based on overlap: for each node, check its suffix
    for node in nodes:
        suffix = node['kmer'][-overlap:]
        candidates = prefix_dict.get(suffix, [])
        for cand_id in candidates:
            # Add an edge from current node to candidate
            edge_dict[(node['id'], cand_id)] = 0

    return edge_dict

def label_groundtruth_edges(edge_dict, gt_order):
    '''
    Label the ground truth edge (the consecutive ordering in gt_order) as 1
    If edge already exists, update its label to 1.
    Otherwise, add it.
    '''
    for i in range(len(gt_order)-1):
        src = gt_order[i]
        tgt = gt_order[i+1]
        edge_dict[(src, tgt)] = 1 #ground truth edge

    return edge_dict

def main():
    if len(sys.argv) != 3:
        print("Usage: python data_preprocessing.py <input_fasta> <output_pickle>")
        sys.exit(1)

    input_fasta = sys.argv[1]
    output_pickle = sys.argv[2]

    all_nodes = []
    all_gt_orders = [] #list of ground truth orders per chromosome
    # Parse the FASTA file - each record is a chromosome

    for record in SeqIO.parse(input_fasta, "fasta"):
        chrom_id = record.id
        sequence = str(record.seq).upper()
        nodes, gt_order = process_chromosome(chrom_id, sequence)
        all_nodes.extend(nodes)
        all_gt_orders.extend(gt_order)
        print(f"Processed chromosome {chrom_id}: {len(nodes)} nodes.")

    #Build the graph based on overlap
    edge_dict = build_graph(all_nodes)
    #label the ground truth edges
    edge_dict = label_groundtruth_edges(edge_dict, all_gt_orders)
    print(f"Total edges (after labeling): {len(edge_dict)}")

    #prepare node features
    node_features = {}
    for node in all_nodes:
        node_features[node['id']] = onhot_encode_kmer(node['kmer'])

    # Convert node_features to a list in the SAME order
    node_id_to_index = {node['id']: idx for idx, node in enumerate(all_nodes)}
    feature_matrix = [node_features[node['id']] for node in all_nodes]
    feature_matrix = np.array(feature_matrix, dtype = np.float32)

    # convert edge_dict to an edge list and label the list
    edges = []
    labels = []
    for (src, tgt), label in edge_dict.items():
        #convert node id strings to indeices.

        if src in node_id_to_index and tgt in node_id_to_index:
            edges.append((node_id_to_index[src], node_id_to_index[tgt]))
            labels.append(label)

    # save graph data
    graph_data = {
        'node_features': feature_matrix, #shape: [num_nodes, feature_dim]
        'edges': edges,                  # list of (src, tgt) tuples
        'edge_labels': labels            # list of binary labels for each edge
    }

    with open(output_pickle, 'wb') as f:
        pickle.dump(graph_data, f)

    print(f"Graph data saved to {output_pickle}")

if __name__ =="__main__":
    main()
