import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

lab_dir = Path(__file__).resolve().parent
plots_dir = lab_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

x = np.genfromtxt(lab_dir / "Train.csv", delimiter=',')
esantioane = x[1:, 0]
valori = x[1:, 2]

start = 1054
stop = start + 30 * 24

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(esantioane[start:stop], valori[start:stop], linewidth=0.8)
ax.set_title("30 de zile de trafic")
ax.set_xlabel("Timp [ore de la inceputul inregistrarii]")
ax.set_ylabel("Numar de masini")
ax.grid(True)

fig.tight_layout()
fig.savefig(plots_dir / "plot_g.pdf", bbox_inches="tight")
plt.close(fig)
