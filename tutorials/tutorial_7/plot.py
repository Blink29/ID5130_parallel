#!/usr/bin/env python3
"""
Plot analytical vs numerical derivatives from MPI finite difference program.
Reads CSV files produced by q1 and generates comparison plots.

Usage: python3 plot.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_derivative(csv_file, dx_label):
    """Plot analytical + numerical derivatives from a single CSV file."""
    data = np.genfromtxt(csv_file, delimiter=',', names=True)

    x = data['x']
    analytical = data['analytical']
    central4   = data['central4']

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

    # Top: overlay analytical and 4th-order numerical
    ax = axes[0]
    ax.plot(x, analytical, 'k-', linewidth=1.5, label='Analytical')
    ax.plot(x, central4, 'r--', linewidth=1.0, alpha=0.8, label='4th-order central')
    ax.set_ylabel("du/dx")
    ax.set_title(f"Derivative of u(x) = x³ − sin(5x),  Δx = {dx_label}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom: error for all schemes
    ax2 = axes[1]
    for col, label, color in [('forward',  '1st-order fwd', 'blue'),
                               ('backward', '1st-order bwd', 'green'),
                               ('central2', '2nd-order central', 'orange'),
                               ('central4', '4th-order central', 'red')]:
        err = np.abs(data[col] - analytical)
        ax2.semilogy(x, err + 1e-18, label=label, alpha=0.8, color=color)

    ax2.set_xlabel("x")
    ax2.set_ylabel("|Error|")
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = csv_file.replace('.csv', '.png')
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()

# Plot for all available CSV files
for fname in sorted(os.listdir('.')):
    if fname.startswith('deriv_dx') and fname.endswith('.csv'):
        dx_val = fname.replace('deriv_dx', '').replace('.csv', '')
        plot_derivative(fname, dx_val)
        
print("Done.")
