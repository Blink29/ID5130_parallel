# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///

import matplotlib.pyplot as plt
import numpy as np
import csv
import sys
import os

csv_file = sys.argv[1] if len(sys.argv) > 1 else "timing_results.csv"

if not os.path.exists(csv_file):
    print(f"Error: {csv_file} not found. Run 'make run' first.")
    sys.exit(1)

N_vals, serial_times, parallel_times = [], [], []

with open(csv_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        N_vals.append(int(row["N"]))
        serial_times.append(float(row["Serial"]))
        parallel_times.append(float(row["Parallel"]))

x = np.arange(len(N_vals))
width = 0.3

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, serial_times, width, label="Serial", color="#2196F3")
bars2 = ax.bar(x + width/2, parallel_times, width, label="Parallel (OpenACC)", color="#FF5722")

ax.set_xlabel("Matrix Size (N)", fontsize=12)
ax.set_ylabel("Time (seconds)", fontsize=12)
ax.set_title("Cholesky Decomposition: Serial vs OpenACC Parallel", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([str(n) for n in N_vals])
ax.legend(fontsize=11)
ax.set_yscale("log")

# Add value labels on bars
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h, f"{h:.4f}s",
            ha="center", va="bottom", fontsize=8)
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h, f"{h:.4f}s",
            ha="center", va="bottom", fontsize=8)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(csv_file) or ".", "cholesky_timing.png")
plt.savefig(out_path, dpi=150)
print(f"Plot saved to {out_path}")
plt.show()
