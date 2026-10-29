import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plots_dir = Path(__file__).resolve().parent / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

def sig(x):
    return 0.8 * np.sin(2 * np.pi * 5 * x) + 1.5 * np.sin(3 * 2 * np.pi * x) + 0.1 * np.cos(32 * 2 * np.pi * x)


N = 100

xs = np.arange(N) / N

F = np.ones((N, N))

i1, i2 = np.indices(F.shape)

F = np.exp(-2 * np.pi * 1j * i1 * i2 / N)

x = sig(xs)

X = F @ x


fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].plot(xs, x)
axs[0].grid(True)

frecvente = np.arange(N //2 + 1)
modul = np.abs(X[:N // 2 + 1])

axs[1].stem(frecvente, modul)
axs[1].grid(True)

plt.tight_layout()
plt.savefig(plots_dir / "plot4.pdf", bbox_inches="tight")
plt.close(fig)

