import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

plots_dir = Path(__file__).resolve().parent / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

PI = np.pi

fs = 1000
I = np.arange(fs) / fs

X = np.sin(50 * 2 * PI * I)

I_dec = I[::4]
X_dec = X[::4]

I_dec2 = I[1::4]
X_dec2 = X[1::4]

fig, axs = plt.subplots(2)

axs[0].plot(I, X, color='red')
axs[0].plot(I_dec, X_dec, 'o-', color='green')
axs[0].set_xlim(0, 0.1)
axs[0].grid(True)

axs[1].plot(I, X, color='red')
axs[1].plot(I_dec2, X_dec2, 'o-', color='blue')
axs[1].set_xlim(0, 0.1)
axs[1].grid(True)

plt.tight_layout()
plt.savefig(plots_dir / "plot7.pdf", bbox_inches="tight")
plt.close(fig)

# a) esantionam in de 4 ori mai putine puncte
# b) esantionul e defazat cu o "pozitie" de cel initial
