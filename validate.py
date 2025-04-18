#!/usr/bin/env python3
import torch
import argparse
from train_model import EdgeClassifier, create_data_object, load_graph_data, split_edge_data, evaluate

parser = argparse.ArgumentParser()
parser.add_argument("pickle_file", help="Graph data pickle")
parser.add_argument("checkpoint",  help="Checkpoint .pt file")
parser.add_argument("--hidden", type=int, default=64, help="Hidden size")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load graph and splits
graph_data = load_graph_data(args.pickle_file)
data       = create_data_object(graph_data)
splits     = split_edge_data(graph_data)

# Rebuild & load model
model = EdgeClassifier(data.num_node_features, args.hidden).to(device)
model.load_state_dict(torch.load(args.checkpoint, map_location=device))
model.eval()

# Evaluate
loss, acc, prec, rec, f1 = evaluate(model, data, splits['test_idx'], splits['test_labels'])
print(f"Test Loss: {loss:.4f}")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
