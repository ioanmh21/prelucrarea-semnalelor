from pathlib import Path


import numpy as np
import matplotlib.pyplot as plt

plots_dir = Path(__file__).resolve().parent / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

PI = np.pi

def x(t):
    return np.sin(2 * PI * 4 * t)

def y(t):
    return np.sin(2 * PI * 11 * t)

def z(t):
    return np.sin(2 * PI * 18 * t)

frecv = 7

I = np.arange(0, frecv + 1) / frecv
I_ = np.arange(0, 2000) / 2000

X = x(I)
X_ = x(I_)
Y = y(I)
Y_ = y(I_)
Z = z(I)
Z_ = z(I_)

fig, axs = plt.subplots(3)

axs[0].plot(I_, X_, color="red", zorder=1)
axs[0].scatter(I, X, color="black", zorder=2)

axs[1].plot(I_, Y_, color="green", zorder=1)
axs[1].scatter(I, Y, color="black", zorder=2)

axs[2].plot(I_, Z_, color="blue", zorder=1)
axs[2].scatter(I, Z, color="black", zorder=2)


plt.tight_layout()
plt.savefig(plots_dir / "plot2.pdf", bbox_inches="tight")
plt.close(fig)

