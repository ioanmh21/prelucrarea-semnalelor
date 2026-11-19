import numpy as np
from pathlib import Path

lab_dir = Path(__file__).resolve().parent

x = np.genfromtxt(lab_dir / "Train.csv", delimiter=',')
valori = x[1:, 2]

valori = valori - np.mean(valori)

N = len(valori)
X = np.fft.fft(valori)
Y = abs(X / len(X))
Y = Y[:N//2]
frecvente = np.fft.fftfreq(N, d=3600)[:N//2]

indici = np.argsort(Y[1:])[-4:][::-1] + 1

for i in indici:
    perioada = 1 / frecvente[i] / 3600
    print(f"f = {frecvente[i]:.8e} Hz, |X[k]| / N = {Y[i]:.4f}, T = {perioada:.2f} ore")

# componenta de 24 ore reflecta ciclul zilnic