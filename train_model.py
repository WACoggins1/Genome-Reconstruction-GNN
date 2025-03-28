'''
Pipeline to:

1. Load the graph data
2. Split the samples into training, validation, and testing.
3. Create a PyTorch Geometric object for the graph.
4. Define an edge classification GNN model that :
    a. computes nodes embeddings via graph convolutions
    b. uses an MLP on concatenated node embeddings for each edge to predict a binary label
5. Train the model and track loss, accuracy, precision, recall, and F1 score.
6. At inference, use a greedy search on predicted edge scores to reconstruct the chromosome.


Usage:
    python train_model.py <graph_data_pickle>

Example:
    python train_model.py graph_data.pkl

'''

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Edge classifier GNN: compute node embedding and classifies each edge.

class EdgeClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(EdgeClassifier, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        # MLP for edge classification
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels,1) #output logit
        )

    def forward(self, x, edge_index):
        #compute edge embeddings
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        #for each edge, get embeddings and source target
        src, tgt = edge_index
        edge_feat = torch.cat([x[src], x[tgt]], dim=1)
        logits = self.edge_mlp(edge_feat).squeeze() #shape: [num_edges]

        return logits


def load_graph_data(pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

    return data

def create_data_object(graph_data):
    '''
    Create a PyTorch Geometric Data object for the full graph
    The entire node feature matrix and edge_index (from edge list) are used.
    '''

    x = torch.tensor(graph_data['node_features'], dtype=torch.float)

    #convert edge list into a tensor of shape [2, num_edges]
    edge_index = torch.tensor(graph_data['edges'], dtype=torch.long).t().contiguous()
    return Data(x = x, edge_index = edge_index)

def split_edge_data(graph_data, test_size = 0.15, val_size = 0.15):
    '''
    Split the edge samples
    Return indices for each split along with their labels
    '''
    edges = np.array(graph_data['edges'])
    labels = np.array(graph_data['edge_labels'])
    num_edges = len(labels)
    indices = np.arange(num_edges)

    #Split off test set
    train_val_idx, test_idx, train_val_labels, test_labels = train_test_split(
        indices, labels, test_size=test_size, random_state=42, stratify=labels)

    #Now split off train_val into training and validation
    # for 5-fold cross validation, loop over folds

    val_fraction = val_size / (1 - test_size)
    train_idx, val_idx, train_labels, val_labels = train_test_split(
        train_val_idx, train_val_labels, test_size=val_fraction, random_state=42, stratify=train_val_labels)

    splits = {
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'train_labels': torch.tensor(train_labels, dtype=torch.float),
        'val_labels': torch.tensor(val_labels, dtype=torch.float),
        'test_labels': torch.tensor(test_labels, dtype=torch.float)
    }
    return splits

def evaluate(model, data, edge_index, true_labels):
    """
    Evaluate the edge classification performance.
    Returns: loss, accuracy, precision, recall, f1.
    """
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        # Select predictions for the specified edge indices.

        true_labels = true_labels.to(device)
        preds = logits[edge_index]
        loss = F.binary_cross_entropy_with_logits(preds, true_labels)
        pred_labels = (torch.sigmoid(preds) >= 0.5).cpu().numpy()
        true_labels_np = true_labels.cpu().numpy()
        acc = accuracy_score(true_labels_np, pred_labels)
        prec = precision_score(true_labels_np, pred_labels, zero_division=0)
        rec = recall_score(true_labels_np, pred_labels, zero_division=0)
        f1 = f1_score(true_labels_np, pred_labels, zero_division=0)
    return loss.item(), acc, prec, rec, f1


def train_model(model, data, splits, num_epochs=100, lr=0.01, weight_decay=1e-4, patience=10):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_loss = float('inf')
    patience_counter = 0

    # Get the full logits (we will index into them using splits).
    full_edge_indices = torch.arange(data.edge_index.size(1))

    # Metrics trackers.
    epochs_list = []
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    train_prec_list = []
    val_prec_list = []
    train_rec_list = []
    val_rec_list = []
    train_f1_list = []
    val_f1_list = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)

        #move the lables to the GPU
        train_labels = splits['train_labels'].to(device)
        train_logits = logits[splits['train_idx'].to(device)]
        train_loss = F.binary_cross_entropy_with_logits(train_logits, splits['train_labels'])
        train_loss.backward()
        optimizer.step()

        # Evaluate training metrics.
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            # Training metrics.
            train_loss_val, train_acc, train_prec, train_rec, train_f1 = evaluate(
                model, data, splits['train_idx'], splits['train_labels'])
            # Validation metrics.
            val_loss_val, val_acc, val_prec, val_rec, val_f1 = evaluate(
                model, data, splits['val_idx'], splits['val_labels'])

        epochs_list.append(epoch)
        train_loss_list.append(train_loss_val)
        val_loss_list.append(val_loss_val)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        train_prec_list.append(train_prec)
        val_prec_list.append(val_prec)
        train_rec_list.append(train_rec)
        val_rec_list.append(val_rec)
        train_f1_list.append(train_f1)
        val_f1_list.append(val_f1)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss_val:.4f}, Val Loss: {val_loss_val:.4f} | "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Checkpoint saving.
        if val_loss_val < best_val_loss:
            best_val_loss = val_loss_val
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_checkpoint.pt")
            print("Checkpoint saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Plot loss and accuracy curves.
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, train_loss_list, label="Train Loss")
    plt.plot(epochs_list, val_loss_list, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_list, train_acc_list, label="Train Accuracy")
    plt.plot(epochs_list, val_acc_list, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Optionally, plot Precision, Recall, and F1 score.
    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    plt.plot(epochs_list, train_prec_list, label="Train Precision")
    plt.plot(epochs_list, val_prec_list, label="Val Precision")
    plt.ylabel("Precision")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(epochs_list, train_rec_list, label="Train Recall")
    plt.plot(epochs_list, val_rec_list, label="Val Recall")
    plt.ylabel("Recall")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(epochs_list, train_f1_list, label="Train F1 Score")
    plt.plot(epochs_list, val_f1_list, label="Val F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return model, {
        'epochs': epochs_list,
        'train_loss': train_loss_list,
        'val_loss': val_loss_list,
        'train_acc': train_acc_list,
        'val_acc': val_acc_list,
        'train_prec': train_prec_list,
        'val_prec': val_prec_list,
        'train_rec': train_rec_list,
        'val_rec': val_rec_list,
        'train_f1': train_f1_list,
        'val_f1': val_f1_list,
    }


def greedy_cycle_reconstruction(data, model):
    """
    Given the full graph and a trained model, reconstruct the Hamiltonian cycle using a greedy search.
    For simplicity, we start from node 0 (assumed to be the first in the ground truth) and at each step select
    the outgoing edge with the highest predicted score that leads to an unvisited node.
    Returns the list of node indices in the predicted cycle.
    """
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        scores = torch.sigmoid(logits).cpu().numpy()

    num_nodes = data.x.size(0)
    # Build a dictionary of outgoing edges: for each source, list (target, score, edge_index)
    out_edges = {}
    edge_index = data.edge_index.cpu().numpy()
    for i in range(edge_index.shape[1]):
        src, tgt = edge_index[:, i]
        out_edges.setdefault(src, []).append((tgt, scores[i], i))

    visited = set()
    cycle = []
    current = 0  # starting node (for demonstration, we assume the ground truth starts at node 0)
    visited.add(current)
    cycle.append(current)

    while len(visited) < num_nodes:
        candidates = out_edges.get(current, [])
        # Filter candidates that lead to unvisited nodes.
        candidates = [c for c in candidates if c[0] not in visited]
        if not candidates:
            print("No candidate edge found. Cycle reconstruction failed.")
            break
        # Greedily select candidate with highest score.
        next_node = max(candidates, key=lambda x: x[1])[0]
        cycle.append(next_node)
        visited.add(next_node)
        current = next_node

    # Optionally, check if we can close the cycle by connecting back to the starting node.
    return cycle


def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python train_gnn.py <graph_data_pickle>")
        sys.exit(1)
    pickle_file = sys.argv[1]
    graph_data = load_graph_data(pickle_file)

    # Create PyTorch Geometric Data object.
    data = create_data_object(graph_data)

    # Split edge data into train/val/test sets.
    splits = split_edge_data(graph_data, test_size=0.15, val_size=0.15)

    # Define model.
    in_channels = data.x.size(1)  # e.g. 31*4 = 124 features
    hidden_channels = 32
    model = EdgeClassifier(in_channels, hidden_channels).to(device) #leverage CUDA

    # Train the model.
    model, metrics = train_model(model, data, splits, num_epochs=100, lr=0.01, weight_decay=1e-4, patience=10)

    # After training, load the best checkpoint.
    model.load_state_dict(torch.load("best_model_checkpoint.pt"))

    # Evaluate on the test set.
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(
        model, data, torch.tensor(splits['test_idx']), splits['test_labels'])
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, "
          f"Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, F1: {test_f1:.4f}")

    # Reconstruct the Hamiltonian cycle using greedy search.
    cycle = greedy_cycle_reconstruction(data, model)
    print("Reconstructed cycle (node indices):")
    print(cycle)


if __name__ == "__main__":
    main()
