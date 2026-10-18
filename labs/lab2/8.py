import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

plots_dir = Path(__file__).resolve().parent / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

PI = np.pi

I = np.linspace(-PI / 2, PI / 2, 1001)

X = np.sin(I)
Y = I
Z = (I - 7 * I ** 3 / 60) / (1 + I ** 2 / 20)

err_taylor = np.abs(X - Y)
err_pade = np.abs(X - Z)

fig, axs = plt.subplots(3, figsize=(7, 8))

axs[0].plot(I, X, color='red')
axs[0].plot(I, Y, color='green')
axs[0].plot(I, Z, '--', color='blue')
axs[0].grid(True)

axs[1].plot(I, err_taylor, color='green')
axs[1].plot(I, err_pade, color='blue')
axs[1].grid(True)

axs[2].plot(I, err_taylor, color='green')
axs[2].plot(I, err_pade, color='blue')
axs[2].set_yscale('log', nonpositive='mask')
axs[2].grid(True)

plt.tight_layout()
plt.savefig(plots_dir / "plot8.pdf", bbox_inches="tight")
plt.close(fig)

