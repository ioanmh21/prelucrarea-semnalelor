from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plots_dir = Path(__file__).resolve().parent / "1_plots"
plots_dir.mkdir(parents=True, exist_ok=True)

PI = np.pi

def x(t : np.double):
    return np.cos(520 * PI * t + PI / 3)

def y(t : np.double):
    return np.cos(280 * PI * t - PI / 3)

def z(t : np.double):
    return np.cos(120 * PI * t + PI / 3)

N = 1001

F = 200

I = np.linspace(0, 0.03, N)

X = x(I)
Y = y(I)
Z = z(I)

fig, axs = plt.subplots(3)

axs[0].plot(I, X, color='red')
axs[0].grid(True)

axs[1].plot(I, Y, color='green')
axs[1].grid(True)

axs[2].plot(I, Z, color='blue')
axs[2].grid(True)

plt.savefig(plots_dir / 'plot1.pdf', bbox_inches='tight')


I_es = np.arange(int(0.03 * F) + 1) / F

axs[0].stem(I_es, x(I_es), markerfmt='.')
axs[1].stem(I_es, y(I_es), markerfmt='.')
axs[2].stem(I_es, z(I_es), markerfmt='.')

plt.savefig(plots_dir / 'plot2.pdf', bbox_inches='tight')

