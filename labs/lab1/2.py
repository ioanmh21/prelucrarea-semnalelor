from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plots_dir = Path(__file__).resolve().parent / "2_plots"
plots_dir.mkdir(parents=True, exist_ok=True)

PI = np.pi

def x(t : np.double):
    return np.sin(400 * 2 * PI * t)

def y(t : np.double):
    return np.sin(800 * 2 * PI * t)

def z(t : np.double):
    return (240 * t) % 1

def a(t : np.double):
    return np.sign(np.sin(300 * 2 * PI * t))



I1 = np.linspace(0, 0.05, 1600)
I2 = np.linspace(0, 3, 100000)
I3 = np.linspace(0, 0.025, 1000)

X = x(I1)
Y = y(I2)
Z = z(I3)
A = a(I3)

fig, axs = plt.subplots(4)

axs[0].plot(I1, X, color='red')
axs[0].grid(True)

axs[1].plot(I2, Y, color='green')
axs[1].set_xlim(0, 0.01)
axs[1].grid(True)

axs[2].plot(I3, Z, color='blue')
axs[2].grid(True)

axs[3].plot(I3, A, color='red')
axs[3].grid(True)

plt.tight_layout()
plt.savefig(plots_dir / "plot1.pdf", bbox_inches="tight")
plt.close(fig)


mat = np.random.rand(128, 128)

mat2 = np.random.rand(128, 128)
mx = np.max(mat2)
mn = np.min(mat2)
mat2 = 2 * ((mat2 - mn) / (mx - mn)) - 1
i, j = np.indices(mat2.shape)
dist_principala = np.abs(i - j)
dist_secundara = np.abs(i + j - 127)
factor = np.maximum(dist_principala, dist_secundara) / 127 
mat2 = factor * np.tan(mat2)

fig, axs = plt.subplots(2, 1, figsize=(5, 10))

axs[0].imshow(mat)

axs[1].imshow(mat2)

plt.tight_layout()
fig.savefig(plots_dir / "plot2.pdf", bbox_inches="tight")
plt.close(fig)
