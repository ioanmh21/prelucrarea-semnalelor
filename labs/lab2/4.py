import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

plots_dir = Path(__file__).resolve().parent / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

PI = np.pi

I = np.linspace(0, 1, 1501)

X = np.sin(3 * 2 * PI * I)
Y = (5 * I) % 1
Z = X + Y

fig, axs = plt.subplots(3)

axs[0].plot(I, X, color='red')
axs[0].grid(True)

axs[1].plot(I, Y, color='green')
axs[1].grid(True)

axs[2].plot(I, Z, color='blue')
axs[2].grid(True)

plt.tight_layout()
plt.savefig(plots_dir / "plot4.pdf", bbox_inches="tight")
plt.close(fig)
