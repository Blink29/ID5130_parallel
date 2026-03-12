import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Read data from q1a_data.txt
# Format: threads time result
with open(os.path.join(script_dir, "q1a_data.txt"), "r") as f:
    lines = f.readlines()

threads = []
times = []
results = []

for line in lines:
    if line.startswith("#") or line.strip() == "":
        continue
    parts = line.strip().split()
    threads.append(int(parts[0]))
    times.append(float(parts[1]))
    results.append(float(parts[2]))

threads = np.array(threads)
times = np.array(times)
serial_time = times[0]

# =============================================
# Plot: Execution Time vs Number of Threads
# =============================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart of times
colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
bars = ax1.bar([str(t) for t in threads], times * 1000, color=colors, edgecolor='black', linewidth=0.5)
ax1.set_xlabel('Number of Threads', fontsize=13)
ax1.set_ylabel('Time (ms)', fontsize=13)
ax1.set_title('Q1a: Riemann Integration - Execution Time\n(Nx=48, Ny=36, Nz=24)', fontsize=13)
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, t in zip(bars, times):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
             f'{t*1000:.3f} ms', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Speedup plot
speedups = serial_time / times
ax2.plot(threads, speedups, 'ro-', linewidth=2, markersize=8, label='Actual Speedup')
ax2.plot(threads, threads, 'b--', linewidth=1.5, alpha=0.5, label='Ideal Speedup')
ax2.set_xlabel('Number of Threads', fontsize=13)
ax2.set_ylabel('Speedup (Serial Time / Parallel Time)', fontsize=13)
ax2.set_title('Q1a: Riemann Integration - Speedup', fontsize=13)
ax2.set_xticks(threads)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'q1a_plot.png'), dpi=150)
plt.close()
print("Saved q1a_plot.png")

# Print summary table
print("\nQ1a Summary (Nx=48, Ny=36, Nz=24):")
print(f"Analytical result = 2040.0")
print(f"{'Threads':<10} {'Time (ms)':<15} {'Result':<15} {'Speedup':<10}")
for i in range(len(threads)):
    sp = serial_time / times[i]
    print(f"{threads[i]:<10} {times[i]*1000:<15.4f} {results[i]:<15.6f} {sp:<10.2f}")
