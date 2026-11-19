import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

lab_dir = Path(__file__).resolve().parent
plots_dir = lab_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

x = np.genfromtxt(lab_dir / "Train.csv", delimiter=',')
esantioane = x[1:, 0]
valori = x[1:, 2]

N = len(valori)
X = np.fft.fft(valori)
frecvente = np.fft.fftfreq(N, d=3600)

#limita inferioara 6 ore
limita = 1 / (6 * 3600)
X_filtrat = X.copy()
X_filtrat[np.abs(frecvente) > limita] = 0

valori_filtrate = np.fft.ifft(X_filtrat).real

start = 1000
stop = start + 30 * 24

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(esantioane[start:stop], valori[start:stop], linewidth=0.8, alpha=0.6, label="Semnal initial")
ax.plot(esantioane[start:stop], valori_filtrate[start:stop], linewidth=1.2, label="Semnal filtrat")
ax.set_title("Eliminarea componentelor de frecventa inalta")
ax.set_xlabel("Timp [ore de la inceputul inregistrarii]")
ax.set_ylabel("Numar de masini")
ax.legend()
ax.grid(True)

fig.tight_layout()
fig.savefig(plots_dir / "plot_i.pdf", bbox_inches="tight")
plt.close(fig)
