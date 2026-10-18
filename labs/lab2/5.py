import numpy as np
import sounddevice

PI = np.pi

fs = 10000
I = np.arange(fs) / fs

X = np.sin(400 * 2 * PI * I)
Y = np.sin(800 * 2 * PI * I)

Z = np.concatenate((X, Y))

sounddevice.play(Z, fs)
sounddevice.wait()

# Y se aude mai ascutit ca X