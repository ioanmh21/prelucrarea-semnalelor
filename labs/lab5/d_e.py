import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

lab_dir = Path(__file__).resolve().parent
plots_dir = lab_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

x = np.genfromtxt(lab_dir / "Train.csv", delimiter=',')

esantioane = x[1:, 0]
valori = x[1:, 2]


N = len(esantioane)
X = np.fft.fft(valori)
Y = abs(X / len(X))
Y = Y[:N//2]
frecvente = np.fft.fftfreq(N, d=3600)[:N//2]

print(Y[0])
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(frecvente, Y, linewidth=0.8)
ax.set_title("Spectrul semnalului")
ax.set_xlabel("Frecventa [Hz]")
ax.set_ylabel("|X[k]| / N")
ax.grid(True)

fig.tight_layout()
fig.savefig(plots_dir / "plot_d.pdf", bbox_inches="tight")
plt.close(fig)

media = np.mean(valori) #componenta continua
valori_fara_medie = valori - media

X = np.fft.fft(valori_fara_medie)
Y = abs(X / len(X))
Y = Y[:N//2]

print(Y[0])
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(frecvente, Y, linewidth=0.8)
ax.set_title("Spectrul semnalului fara componenta continua")
ax.set_xlabel("Frecventa [Hz]")
ax.set_ylabel("|X[k]| / N")
ax.grid(True)

fig.tight_layout()
fig.savefig(plots_dir / "plot_e.pdf", bbox_inches="tight")
plt.close(fig)
