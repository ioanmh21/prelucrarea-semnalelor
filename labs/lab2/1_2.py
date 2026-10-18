import scipy
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

plots_dir = Path(__file__).resolve().parent / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)


PI = np.pi

def x(t : np.double, phi : np.double = 0):
    return np.sin(3 * 2 * PI * t + PI / 5 + phi)

def y(t : np.double):
    return np.cos(3 * 2 * PI * t + PI / 5 - PI / 2)

I = np.linspace(0, 1, 1501)

X = x(I)
Y = y(I)

fig, axs = plt.subplots(2)

axs[0].plot(I, X, color='red')
axs[0].grid(True)

axs[1].plot(I, Y, color='green')
axs[1].grid(True)

plt.savefig(plots_dir / 'plot1.pdf', bbox_inches='tight')


l = np.array([x(I, phi) for phi in range(4)])

plt.figure()
plt.plot(I, l.T)
#l.T pentru a plota fiecare unda din l separat
plt.grid(True)
plt.savefig(plots_dir / "plot2_1.pdf", bbox_inches="tight")
plt.close()


new_l = []
x_norm = np.linalg.norm(l[0])

for ratio in [0.1, 1, 10, 100]:
    z = np.random.normal(size=np.size(l[0]))
    z_norm = np.linalg.norm(z)

    y = np.sqrt(x_norm * x_norm / (ratio * z_norm * z_norm))
    X_noise = l[0] + y * z
    new_l.append(X_noise)
new_l = np.array(new_l)

plt.figure()
plt.plot(I, new_l.T)
plt.grid(True)
plt.savefig(plots_dir / "plot2_2.pdf", bbox_inches="tight")
plt.close()