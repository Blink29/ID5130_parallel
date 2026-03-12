import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# =============================================
# Parse serial data
# =============================================
with open(os.path.join(script_dir, "q2c_serial_data.txt"), "r") as f:
    serial_lines = f.readlines()

idx = 0
assert serial_lines[idx].strip() == "VERIFICATION"
idx += 1
n_verify = int(serial_lines[idx].strip())
idx += 1

x_serial = []
phi_serial = []
for i in range(n_verify):
    parts = serial_lines[idx].strip().split()
    x_serial.append(float(parts[0]))
    phi_serial.append(float(parts[1]))
    idx += 1

assert serial_lines[idx].strip() == "TIMING"
idx += 1
n_timing = int(serial_lines[idx].strip())
idx += 1

deltas_s = []
grids_s = []
times_s = []
for i in range(n_timing):
    parts = serial_lines[idx].strip().split()
    deltas_s.append(float(parts[0]))
    grids_s.append(int(parts[1]))
    times_s.append(float(parts[2]))
    idx += 1

# =============================================
# Parse parallel data
# =============================================
with open(os.path.join(script_dir, "q2c_parallel_data.txt"), "r") as f:
    par_lines = f.readlines()

idx = 0
assert par_lines[idx].strip() == "VERIFICATION"
idx += 1
n_verify_p = int(par_lines[idx].strip())
idx += 1

x_par = []
phi_par = []
for i in range(n_verify_p):
    parts = par_lines[idx].strip().split()
    x_par.append(float(parts[0]))
    phi_par.append(float(parts[1]))
    idx += 1

assert par_lines[idx].strip() == "TIMING"
idx += 1
n_timing_p = int(par_lines[idx].strip())
idx += 1

deltas_p = []
grids_p = []
times_p = []
for i in range(n_timing_p):
    parts = par_lines[idx].strip().split()
    deltas_p.append(float(parts[0]))
    grids_p.append(int(parts[1]))
    times_p.append(float(parts[2]))
    idx += 1

# =============================================
# Plot 1: Verification - Serial vs Parallel at y=0.5 (Δ=0.1)
# =============================================
plt.figure(figsize=(8, 6))
plt.plot(x_serial, phi_serial, 'b-', linewidth=2, label='Serial')
plt.plot(x_par, phi_par, 'ro--', markersize=6, label='Parallel (8 threads)')
plt.xlabel('x', fontsize=14)
plt.ylabel(r'$\phi$', fontsize=14)
plt.title(r'Verification: Serial vs Parallel $\phi$ at $y=0.5$ ($\Delta=0.1$)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'q2c_verification.png'), dpi=150)
plt.close()
print("Saved q2c_verification.png")

# =============================================
# Plot 2: Time vs Δ for serial and parallel
# =============================================
plt.figure(figsize=(8, 6))
plt.plot(deltas_s, times_s, 'bs-', linewidth=2, markersize=8, label='Serial')
plt.plot(deltas_p, times_p, 'ro-', linewidth=2, markersize=8, label='Parallel (8 threads)')
plt.xlabel(r'$\Delta$', fontsize=14)
plt.ylabel('Time (seconds)', fontsize=14)
plt.title(r'Time vs $\Delta$: Serial vs Parallel (8 threads)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.gca().invert_xaxis()  # smaller Δ = finer grid = more work
plt.yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'q2c_timing.png'), dpi=150)
plt.close()
print("Saved q2c_timing.png")

# Print summary
print("\nTiming Summary:")
print(f"{'Delta':<10} {'Grid N':<10} {'Serial (s)':<15} {'Parallel (s)':<15} {'Speedup':<10}")
for i in range(n_timing):
    speedup = times_s[i] / times_p[i] if times_p[i] > 0 else 0
    print(f"{deltas_s[i]:<10.4f} {grids_s[i]:<10} {times_s[i]:<15.4f} {times_p[i]:<15.4f} {speedup:<10.2f}")
