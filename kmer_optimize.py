import os
from collections import defaultdict, Counter
from Bio import SeqIO
import matplotlib.pyplot as plt
import numpy as np

# Parameters
input_fasta = "GCF_000005845.2_ASM584v2_genomic.fna"
k_values = range(21, 102, 10)
min_abundance_threshold = 2  # filter out singletons (likely errors)
sample_limit = None  # Set to an int if you want to limit the number of reads for testing

# Store results for each k
kmer_stats = {}

for k in k_values:
    print(f"Processing k = {k}")
    kmer_counts = Counter()

    for record in SeqIO.parse(input_fasta, "fasta"):
        seq = str(record.seq).upper()
        if 'N' in seq:
            continue  # skip ambiguous reads
        if sample_limit and len(kmer_counts) > sample_limit:
            break
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            kmer_counts[kmer] += 1

    abundance_histogram = defaultdict(int)
    for count in kmer_counts.values():
        abundance_histogram[count] += 1

    # Estimate "genomic" k-mers = those with abundance >= threshold
    genomic_kmers = sum(v for abun, v in abundance_histogram.items() if abun >= min_abundance_threshold)
    total_kmers = sum(abundance_histogram.values())

    kmer_stats[k] = {
        "genomic_kmers": genomic_kmers,
        "total_kmers": total_kmers,
        "histogram": abundance_histogram
    }

# Select k with max genomic_kmers
optimal_k = max(kmer_stats.items(), key=lambda x: x[1]["genomic_kmers"])[0]
print(f"Optimal k-mer size based on abundance filtering: {optimal_k}")

# Optional: Plot
plt.figure(figsize=(10, 6))
k_vals = list(kmer_stats.keys())
genomic_counts = [kmer_stats[k]["genomic_kmers"] for k in k_vals]
plt.plot(k_vals, genomic_counts, marker='o')
plt.xlabel("k-mer size")
plt.ylabel("Estimated Genomic k-mers")
plt.title("Optimal k-mer size estimation")
plt.grid(True)
plt.savefig("kmer_genomic_estimate.png")
plt.show()

# Save result
with open("optimal_k.txt", "w") as f:
    f.write(str(optimal_k))
