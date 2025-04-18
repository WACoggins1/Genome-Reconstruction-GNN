#!/usr/bin/env python
"""
eda_extended.py

This script performs extended exploratory data analysis (EDA) on a FASTA file.
It computes basic sequence statistics, k-mer frequency analysis, and a sliding window GC content plot.

Usage:
    python eda_extended.py <input_fasta> [k] [window_size] [step_size]

Defaults:
    k = 31
    window_size = 1000 bp
    step_size = 100 bp
"""

import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Bio import SeqIO


def calculate_nucleotide_composition(seq):
    counts = {nuc: seq.count(nuc) for nuc in "ACGT"}
    total = sum(counts.values())
    composition = {nuc: count / total for nuc, count in counts.items()}
    return counts, composition


def calculate_gc_content(seq):
    gc = seq.count("G") + seq.count("C")
    return gc / len(seq) if len(seq) > 0 else 0


def compute_kmer_freq(seq, k):
    """
    Compute the frequency of each k-mer in the sequence.

    Args:
        seq (str): DNA sequence.
        k (int): k-mer length.

    Returns:
        dict: keys are k-mers and values are frequencies.
    """
    freq = {}
    n = len(seq)
    for i in range(n - k + 1):
        kmer = seq[i:i + k]
        freq[kmer] = freq.get(kmer, 0) + 1
    return freq


def sliding_window_gc(seq, window_size, step_size):
    """
    Compute the GC content using a sliding window.

    Args:
        seq (str): DNA sequence.
        window_size (int): Window length.
        step_size (int): Step size for the window.

    Returns:
        positions (list): Starting position of each window.
        gc_contents (list): GC content for each window.
    """
    positions = []
    gc_contents = []
    for i in range(0, len(seq) - window_size + 1, step_size):
        window = seq[i:i + window_size]
        gc = calculate_gc_content(window)
        positions.append(i)
        gc_contents.append(gc)
    return positions, gc_contents


def main(input_fasta, k=31, window_size=1000, step_size=100):
    records = list(SeqIO.parse(input_fasta, "fasta"))
    summary_data = []

    for record in records:
        seq = str(record.seq).upper()
        length = len(seq)
        counts, composition = calculate_nucleotide_composition(seq)
        overall_gc = calculate_gc_content(seq)
        summary_data.append({
            "id": record.id,
            "length": length,
            "A": counts["A"],
            "C": counts["C"],
            "G": counts["G"],
            "T": counts["T"],
            "GC_content": overall_gc
        })

        # --- Plot K-mer Frequency Analysis ---
        print(f"Processing k-mer frequency for {record.id} ...")
        kmer_freq = compute_kmer_freq(seq, k)
        freq_values = list(kmer_freq.values())

        plt.figure(figsize=(8, 5))
        plt.hist(freq_values, bins=50, edgecolor="black")
        plt.xlabel("K-mer Frequency")
        plt.ylabel("Count")
        plt.title(f"K-mer (k={k}) Frequency Distribution for {record.id}")
        plt.yscale("log")  # Use a log scale to better visualize the distribution
        plt.show()


        # --- Plot Sliding Window GC Content ---
        print(f"Processing sliding window GC content for {record.id} ...")
        positions, gc_values = sliding_window_gc(seq, window_size, step_size)

        plt.figure(figsize=(10, 5))
        plt.plot(positions, gc_values, marker='o', linestyle='-')
        plt.xlabel("Position")
        plt.ylabel("GC Content")
        plt.title(f"Sliding Window GC Content (window={window_size}, step={step_size}) for {record.id}")
        plt.ylim(0, 1)
        plt.show()

    # Create and display a summary DataFrame.
    df = pd.DataFrame(summary_data)
    print("Sequence Summary Statistics:")
    print(df.describe())

    # Plot distribution of sequence lengths if there is more than one sequence.
    if len(df) > 1:
        plt.figure(figsize=(8, 5))
        plt.hist(df["length"], bins=20, edgecolor="black")
        plt.xlabel("Sequence Length")
        plt.ylabel("Frequency")
        plt.title("Distribution of Sequence Lengths")
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.hist(df["GC_content"], bins=20, edgecolor="black")
        plt.xlabel("GC Content")
        plt.ylabel("Frequency")
        plt.title("Distribution of GC Content")
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eda_extended.py <input_fasta> [k] [window_size] [step_size]")
        sys.exit(1)

    input_fasta = sys.argv[1]
    # Get optional parameters or use defaults.
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 31
    window_size = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    step_size = int(sys.argv[4]) if len(sys.argv) > 4 else 100

    main(input_fasta, k, window_size, step_size)
