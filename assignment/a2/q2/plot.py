#!/usr/bin/env python3
# /// script
# dependencies = ["numpy", "matplotlib"]
# ///
"""Plot Poisson solutions: φ vs x at y=0 and φ vs y at x=0"""

import numpy as np
import matplotlib.pyplot as plt
import os, glob

def load(f):
    d = np.genfromtxt(f, delimiter=',', skip_header=1)
    return d[:,0], d[:,1], d[:,2]

# --- Part (a): Serial delta=0.1 ---
f = "q2a_d0.100000_data.csv"
if os.path.exists(f):
    c, px, py = load(f)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Poisson Eq - Serial Jacobi (Δ=0.1)", fontweight='bold')
    ax1.plot(c, px, 'b-o', markersize=3, label='Jacobi')
    ax1.set(xlabel='x', ylabel='φ', title='φ vs x at y=0'); ax1.grid(True, alpha=0.3); ax1.legend()
    ax2.plot(c, py, 'r-o', markersize=3, label='Jacobi')
    ax2.set(xlabel='y', ylabel='φ', title='φ vs y at x=0'); ax2.grid(True, alpha=0.3); ax2.legend()
    plt.tight_layout(); plt.savefig("q2a_plot.png", dpi=150); plt.close()
    print("Saved q2a_plot.png")

# --- Part (b): Compare serial d=0.01 with MPI p=2,4,8 ---
serial_f = "q2a_d0.010000_data.csv"
mpi_files = sorted(glob.glob("q2b_p*_data.csv"))
if os.path.exists(serial_f) or mpi_files:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Poisson Eq - MPI Jacobi (Δ=0.01)", fontweight='bold')
    if os.path.exists(serial_f):
        c, px, py = load(serial_f)
        ax1.plot(c, px, 'k-', lw=2, label='Serial')
        ax2.plot(c, py, 'k-', lw=2, label='Serial')
    styles = ['b--', 'r-.', 'g:']
    for i, mf in enumerate(mpi_files):
        c, px, py = load(mf)
        p = mf.split('_p')[1].split('_')[0]
        ax1.plot(c, px, styles[i%3], lw=1.2, label=f'MPI p={p}')
        ax2.plot(c, py, styles[i%3], lw=1.2, label=f'MPI p={p}')
    ax1.set(xlabel='x', ylabel='φ', title='φ vs x at y=0'); ax1.grid(True, alpha=0.3); ax1.legend()
    ax2.set(xlabel='y', ylabel='φ', title='φ vs y at x=0'); ax2.grid(True, alpha=0.3); ax2.legend()
    plt.tight_layout(); plt.savefig("q2b_plot.png", dpi=150); plt.close()
    print("Saved q2b_plot.png")
