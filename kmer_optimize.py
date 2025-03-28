import subprocess
import os

# Path to the input FASTA file
fasta_file = "input_reads.fasta"

# Path to the KmerGenie executable (ensure it's installed and accessible)
kmergenie_exec = "/path/to/kmergenie"

# Directory to store KmerGenie results
output_dir = "kmergenie_output"
os.makedirs(output_dir, exist_ok=True)

# Run KmerGenie to determine the optimal k-mer size
subprocess.run([
    kmergenie_exec,
    fasta_file,
    "-o", output_dir
], check=True)

# Parse the optimal k-mer size from KmerGenie's report
optimal_k = None
report_file = os.path.join(output_dir, "best_k.txt")

with open(report_file, 'r') as f:
    optimal_k = int(f.readline().strip())

print(f"Optimal k-mer size determined by KmerGenie: {optimal_k}")

# Now you can proceed with sampling or assembly using the optimal k
# This value can be passed downstream to your sampling and assembly scripts

# Example usage in the pipeline:
with open("optimal_k.txt", "w") as f:
    f.write(str(optimal_k))
