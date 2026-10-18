from scipy.io import wavfile
import numpy as np
import sounddevice

from pathlib import Path

PI = np.pi

fs = 10000
I = np.arange(fs) / fs  

X = np.sin(400 * 2 * PI * I)
Y = np.sin(800 * 2 * PI * I)
Z = (240 * I) % 1
A = np.sign(np.sin(300 * 2 * PI * I))


for semnal in [X, Y, Z, A]:
    sounddevice.play(semnal, fs)
    sounddevice.wait()

wav_path = Path(__file__).resolve().parent / "semnal.wav"
wavfile.write(wav_path, fs, X.astype(np.float32))


fs_loaded, X_loaded = wavfile.read(wav_path)

sounddevice.play(X_loaded, fs_loaded)
sounddevice.wait()

