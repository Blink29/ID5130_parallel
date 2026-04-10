import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("q3_results.dat")
x     = data[:, 0]
exact = data[:, 1]
pade  = data[:, 2]
cds2  = data[:, 3]

# Fine grid for smooth exact curve
xf = np.linspace(0, 3, 500)
exact_fine = 5.0 * np.cos(5.0 * xf)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Derivative comparison ---
ax1.plot(xf, exact_fine, 'k-', lw=1.5, label='Exact (5 cos 5x)')
ax1.plot(x, pade, 'ro-', ms=6, lw=1.2, label='Padé 4th-order')
ax1.plot(x, cds2, 'bs--', ms=6, lw=1.2, label='CDS2 + FD1/BD1')
ax1.set_xlabel('x')
ax1.set_ylabel("f'(x)")
ax1.set_title("Derivative of f(x) = sin(5x)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Error comparison ---
ax2.plot(x, np.abs(exact - pade), 'ro-', ms=6, lw=1.2, label='Padé 4th-order')
ax2.plot(x, np.abs(exact - cds2), 'bs--', ms=6, lw=1.2, label='CDS2 + FD1/BD1')
ax2.set_xlabel('x')
ax2.set_ylabel('|Error|')
ax2.set_title('Absolute Error')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig("q3_plot.png", dpi=150)
plt.show()
print("Plot saved to q3_plot.png")
