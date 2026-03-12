import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Read data
with open(os.path.join(script_dir, "q2a_data.txt"), "r") as f:
    lines = f.readlines()

iterations = int(lines[0].strip())
print(f"Iterations to converge: {iterations}")

x_vals = []
phi_num = []
phi_exact = []

for line in lines[1:]:
    parts = line.strip().split()
    if len(parts) == 3:
        x_vals.append(float(parts[0]))
        phi_num.append(float(parts[1]))
        phi_exact.append(float(parts[2]))

x_vals = np.array(x_vals)
phi_num = np.array(phi_num)
phi_exact = np.array(phi_exact)

plt.figure(figsize=(8, 6))
plt.plot(x_vals, phi_exact, 'b-', linewidth=2, label='Exact Solution')
plt.plot(x_vals, phi_num, 'ro--', markersize=6, label=f'Numerical (Jacobi, {iterations} iters)')
plt.xlabel('x', fontsize=14)
plt.ylabel(r'$\phi$', fontsize=14)
plt.title(r'$\phi$ vs $x$ at $y = 0.5$ ($\Delta = 0.1$)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'q2a_plot.png'), dpi=150)
plt.close()
print("Plot saved to q2a_plot.png")
