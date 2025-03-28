#!/usr/bin/env python
"""
sample_fasta_multiple.py

This script samples multiple small subsets from each chromosome in a large FASTA file,
creating a new FASTA that is manageable for testing or prototyping.

Usage:
    python sample_fasta_multiple.py <input_fasta> <output_fasta> <max_bases> <num_samples> [--seed <seed>]

Example:
    python sample_fasta_multiple.py GCF_000001405.40_GRCh38.p14_genomic.fna sampled_genome.fasta 100000 5 --seed 42

This will sample 5 regions, each of 100,000 bases (100 kb), from each chromosome.
"""

import sys
import random
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

def sample_fasta(input_fasta, output_fasta, max_bases, num_samples=1, seed=None):
    """
    For each record in the input FASTA, sample a random contiguous subset of length max_bases.
    Do this num_samples times per record.
    Write the sampled sequences to output_fasta.
    
    Args:
        input_fasta (str): Path to the input FASTA file.
        output_fasta (str): Path to the output FASTA file.
        max_bases (int): Maximum number of bases for each sampled region.
        num_samples (int): Number of samples to take from each record.
        seed (int, optional): Random seed for reproducibility.
    """
    if seed is not None:
        random.seed(seed)
    
    records_to_write = []
    for record in SeqIO.parse(input_fasta, "fasta"):
        chrom_len = len(record.seq)
        if chrom_len <= max_bases:
            # If the record is already short, write it once.
            records_to_write.append(record)
        else:
            for sample_num in range(num_samples):
                start_pos = random.randint(0, chrom_len - max_bases)
                end_pos = start_pos + max_bases
                subset_seq = record.seq[start_pos:end_pos]
                new_record_id = f"{record.id}_sample{sample_num+1}_{start_pos}_{end_pos}"
                new_record = SeqRecord(
                    Seq(str(subset_seq)),
                    id=new_record_id,
                    description=f"Sampled from {record.id}, pos={start_pos}-{end_pos}"
                )
                records_to_write.append(new_record)
    
    SeqIO.write(records_to_write, output_fasta, "fasta")
    print(f"Done. Wrote {len(records_to_write)} sampled records to {output_fasta}.")

def main():
    if len(sys.argv) < 5:
        print("Usage: python sample_fasta_multiple.py <input_fasta> <output_fasta> <max_bases> <num_samples> [--seed <seed>]")
        sys.exit(1)
    
    input_fasta = sys.argv[1]
    output_fasta = sys.argv[2]
    max_bases = int(sys.argv[3])
    num_samples = int(sys.argv[4])
    
    seed = None
    if "--seed" in sys.argv:
        seed_index = sys.argv.index("--seed")
        if seed_index + 1 < len(sys.argv):
            seed = int(sys.argv[seed_index + 1])
    
    sample_fasta(input_fasta, output_fasta, max_bases, num_samples, seed)

if __name__ == "__main__":
    main()
