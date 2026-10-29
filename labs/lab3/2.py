import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

plots_dir = Path(__file__).resolve().parent / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)


X = np.linspace(0, 1, 5000)

def sin(x, frecv = 7):
    return np.sin(2 * np.pi * frecv * x)

Y = sin(X) * np.exp(-2 * np.pi * 1j * X)

Z = [sin(X) * np.exp(-2 * np.pi * 1j * frecv * X) for frecv in (0.5, 2, 4, 7)]

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].plot(X, sin(X))
axs[0].grid(True)

axs[1].scatter(Y.real, Y.imag, c=np.abs(Y), cmap='viridis', vmin=0, vmax=1, s=5)
axs[1].set_aspect('equal')
axs[1].grid(True)

plt.tight_layout()
plt.savefig(plots_dir / "plot2.pdf", bbox_inches="tight")
plt.close(fig)


fig, axs = plt.subplots(2, 2, figsize=(12, 12))
axs = axs.flatten()
for i in range(4):
    axs[i].scatter(Z[i].real, Z[i].imag, c=np.abs(Z[i]), cmap='viridis', vmin=0, vmax=1, s=5)
    axs[i].set_aspect('equal')
    axs[i].grid(True)

plt.tight_layout()
plt.savefig(plots_dir / "plot3.pdf", bbox_inches="tight")
plt.close(fig)





fig, axs = plt.subplots(3, 2, figsize=(10, 12))
axs = axs.flatten()

semnal = sin(X)
curbe = [Y] + Z
coordonate = [np.column_stack((z.real, z.imag)) for z in curbe]
culori = [np.abs(z) for z in curbe]

for ax in axs:
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True)

axs[0].set_xlim(0, 1)
linie, = axs[0].plot([], [])
indicator_semnal, = axs[0].plot([], [], 'ro')

puncte = []
indicatori = []
for ax in axs[1:]:
    ax.set_xlim(-1.1, 1.1)
    ax.set_aspect('equal')
    puncte.append(ax.scatter([], [], c=[], cmap='viridis',
                             vmin=0, vmax=1, s=5, linewidths=0))
    indicator, = ax.plot([], [], 'ro')
    indicatori.append(indicator)

plt.tight_layout()

def actualizeaza(n):
    linie.set_data(X[:n], semnal[:n])
    indicator_semnal.set_data([X[n - 1]], [semnal[n - 1]])

    for i in range(len(curbe)):
        puncte[i].set_offsets(coordonate[i][:n])
        puncte[i].set_array(culori[i][:n])
        indicatori[i].set_data([curbe[i][n - 1].real], [curbe[i][n - 1].imag])

    return [linie, indicator_semnal] + puncte + indicatori

cadre = list(range(1, len(X), 10)) + [len(X)]
animatie = FuncAnimation(fig, actualizeaza, frames=cadre, interval=16,
                        blit=fig.canvas.supports_blit, repeat=False)
plt.show()
