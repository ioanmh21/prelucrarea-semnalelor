import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plots_dir = Path(__file__).resolve().parent / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)



N = 8

F = np.ones((8, 8))

i1, i2 = np.indices(F.shape)

F = np.exp(-2 * np.pi * 1j * i1 * i2 / N)

fig, axs = plt.subplots(N, figsize=(8, 12), sharex=True)

for i in range(N):
    axs[i].stem(F[i].real, linefmt='r-', markerfmt='ro', basefmt='k-')
    axs[i].stem(F[i].imag, linefmt='b--', markerfmt='bo', basefmt='k-')
    axs[i].set_ylabel(f"Linia {i}")
    axs[i].set_ylim(-1.2, 1.2)
    axs[i].grid(True)


plt.tight_layout()
plt.savefig(plots_dir / "plot1.pdf", bbox_inches="tight")
plt.close(fig)


F_h = np.conjugate(F)
I = F_h @ F
print(np.allclose((1 / I[0][0]) * I,  np.identity(N)))