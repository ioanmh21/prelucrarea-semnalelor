import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

plots_dir = Path(__file__).resolve().parent / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

PI = np.pi

fs = 15
I = np.arange(fs) / fs

X = np.sin((fs / 2) * 2 * PI * I)
Y = np.sin((fs / 4) * 2 * PI * I)
Z = np.sin(0 * 2 * PI * I)

fig, axs = plt.subplots(3)

axs[0].plot(I, X, 'o-', color='red')
axs[0].set_title("f = fs / 2")
axs[0].set_ylim(-1.1, 1.1)
axs[0].grid(True)

axs[1].plot(I, Y, 'o-', color='green')
axs[1].set_title("f = fs / 4")
axs[1].set_ylim(-1.1, 1.1)
axs[1].grid(True)

axs[2].plot(I, Z, 'o-', color='blue')
axs[2].set_title("f = 0 Hz")
axs[2].set_ylim(-1.1, 1.1)
axs[2].grid(True)

plt.tight_layout()
plt.savefig(plots_dir / "plot6.pdf", bbox_inches="tight")
plt.close(fig)

# in fs/2 avem doar 0 pentru ca esantionam in multipli de PI
# in fs/4 esantionam in multiplii lui PI/2 deci e o secventa 0, 1, 0, -1, ...

