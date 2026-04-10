#!/usr/bin/env python3
# /// script
# dependencies = ["numpy", "matplotlib"]
# ///
"""Plot wave equation solutions: analytical vs upwind vs QUICK at t=0, 0.5, 1.0"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def plot_results(csv_file, title_suffix=""):
    data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
    x = data[:, 0]
    t_vals = [0.0, 0.5, 1.0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"1D Wave Equation{title_suffix}", fontsize=14, fontweight='bold')

    for k, t in enumerate(t_vals):
        col = 1 + k * 3
        exact  = data[:, col]
        upwind = data[:, col + 1]
        quick  = data[:, col + 2]

        ax = axes[k]
        ax.plot(x, exact,  'k-',  linewidth=1.5, label='Exact')
        ax.plot(x, upwind, 'b--', linewidth=1.0, label='Upwind')
        ax.plot(x, quick,  'r-.', linewidth=1.0, label='QUICK')
        ax.set_title(f't = {t}')
        ax.set_xlabel('x')
        ax.set_ylabel('u(x,t)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 2])
        ax.set_ylim([-1.2, 1.2])

    plt.tight_layout()
    outname = csv_file.replace('.csv', '.png')
    plt.savefig(outname, dpi=150)
    print(f"Plot saved to {outname}")
    plt.close()

if __name__ == '__main__':
    files = sys.argv[1:] if len(sys.argv) > 1 else ['q1a_data.csv']
    for f in files:
        if os.path.exists(f):
            suffix = ""
            if "p2" in f: suffix = " (MPI p=2)"
            elif "p4" in f: suffix = " (MPI p=4)"
            elif "q1a" in f: suffix = " (Serial)"
            plot_results(f, suffix)
        else:
            print(f"File {f} not found")
