from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


plots_dir = Path(__file__).resolve().parent / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)


frecv, semnal = wavfile.read(
    Path(__file__).resolve().parent / "aeiou.wav"
)

if semnal.ndim == 2:
    semnal = semnal.mean(axis=1)

N = len(semnal)
lungime = N // 100
pas = lungime // 2

spectre = []

for i in range(0, N - lungime + 1, pas):
    grup = semnal[i:i + lungime]

    spectru = np.abs(np.fft.rfft(grup))
    spectre.append(spectru)

matrice = np.column_stack(spectre)

matrice_db = 20 * np.log10(matrice + 1e-10)

maxim = matrice_db.max()

plt.imshow(
    matrice_db,
    cmap="inferno",
    origin="lower",
    aspect="auto",
    extent=[0, N / frecv, 0, frecv / 2],
    vmin=maxim - 80,
    vmax=maxim
)

plt.ylim(0, 5000)
plt.tight_layout()
plt.savefig(plots_dir / "plot6.pdf", bbox_inches="tight")
plt.show()
