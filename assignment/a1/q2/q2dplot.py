import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(script_dir, "q2d_data.txt"), "r") as f:
    lines = f.readlines()

threads = []
times = []

for line in lines:
    parts = line.strip().split()
    if len(parts) == 2:
        threads.append(int(parts[0]))
        times.append(float(parts[1]))

threads = np.array(threads)
times = np.array(times)

# Separate serial (threads==1) and parallel
serial_time = times[0]  # threads=1 is serial

plt.figure(figsize=(8, 6))
plt.plot(threads, times, 'bo-', linewidth=2, markersize=8, label='Measured Time')
plt.axhline(y=serial_time, color='r', linestyle='--', linewidth=1.5, label=f'Serial Time ({serial_time:.2f}s)')
plt.xlabel('Number of Threads', fontsize=14)
plt.ylabel('Time (seconds)', fontsize=14)
plt.title(r'Time vs Number of Threads ($\Delta = 0.005$)', fontsize=14)
plt.xticks(threads)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'q2d_plot.png'), dpi=150)
plt.close()
print("Saved q2d_plot.png")

# Print summary
print("\nThread Scaling Summary (delta=0.005):")
print(f"{'Threads':<10} {'Time (s)':<15} {'Speedup':<10}")
for i in range(len(threads)):
    speedup = serial_time / times[i] if times[i] > 0 else 0
    print(f"{threads[i]:<10} {times[i]:<15.4f} {speedup:<10.2f}")
