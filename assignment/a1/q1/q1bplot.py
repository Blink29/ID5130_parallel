import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Read data from q1b_data.txt
# Format: N threads time result rel_error
with open(os.path.join(script_dir, "q1b_data.txt"), "r") as f:
    lines = f.readlines()

data = []
for line in lines:
    if line.startswith("#") or line.strip() == "":
        continue
    parts = line.strip().split()
    data.append({
        'N': int(parts[0]),
        'threads': int(parts[1]),
        'time': float(parts[2]),
        'result': float(parts[3]),
        'rel_error': float(parts[4])
    })

N_values = sorted(set(d['N'] for d in data))
thread_counts = sorted(set(d['threads'] for d in data))

# Build time matrix: rows=N, cols=threads
time_matrix = {}
for d in data:
    time_matrix[(d['N'], d['threads'])] = d['time']

error_matrix = {}
for d in data:
    error_matrix[(d['N'], d['threads'])] = d['rel_error']

# =============================================
# Plot 1: Execution Time vs N for each thread count
# =============================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

markers = ['s', 'o', '^', 'D']
colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

for i, t in enumerate(thread_counts):
    label = 'Serial' if t == 1 else f'{t} threads'
    times_for_t = [time_matrix[(N, t)] for N in N_values]
    ax1.plot(N_values, times_for_t, marker=markers[i], color=colors[i],
             linewidth=2, markersize=7, label=label)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('N (Number of Sample Points)', fontsize=13)
ax1.set_ylabel('Time (seconds)', fontsize=13)
ax1.set_title('Q1b: Monte Carlo Integration - Execution Time', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3, which='both')

# =============================================
# Plot 2: Speedup vs Threads for large N values
# =============================================
# Only plot speedup for larger N where parallelism actually helps
large_N = [N for N in N_values if N >= 10000]

for i, N in enumerate(large_N):
    serial_time = time_matrix[(N, 1)]
    speedups = [serial_time / time_matrix[(N, t)] for t in thread_counts]
    ax2.plot(thread_counts, speedups, marker=markers[i], linewidth=2,
             markersize=7, label=f'N={N:,}')

ax2.plot(thread_counts, thread_counts, 'k--', linewidth=1.5, alpha=0.4, label='Ideal')
ax2.set_xlabel('Number of Threads', fontsize=13)
ax2.set_ylabel('Speedup', fontsize=13)
ax2.set_title('Q1b: Monte Carlo Integration - Speedup', fontsize=13)
ax2.set_xticks(thread_counts)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'q1b_plot.png'), dpi=150)
plt.close()
print("Saved q1b_plot.png")

# =============================================
# Print summary tables
# =============================================
print("\nQ1b Timing Summary:")
print(f"{'N':<12} {'Threads':<10} {'Time (s)':<15} {'Result':<12} {'Rel Err %':<12} {'Speedup':<10}")
for N in N_values:
    serial_time = time_matrix[(N, 1)]
    for t in thread_counts:
        sp = serial_time / time_matrix[(N, t)]
        d = [x for x in data if x['N'] == N and x['threads'] == t][0]
        print(f"{N:<12} {t:<10} {d['time']:<15.6f} {d['result']:<12.2f} {d['rel_error']:<12.4f} {sp:<10.2f}")
