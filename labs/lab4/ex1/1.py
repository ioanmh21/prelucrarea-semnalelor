#!/usr/bin/env python3
"""Exercitiul 1: comparatie DFT proprie, FFT C++ si numpy.fft.

Rulare din radacina proiectului:
    ./.venv/bin/python labs/lab4/ex1/1.py

Graficul este salvat in labs/lab4/ex1/plots/plot1.pdf.
"""

import argparse
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-lab4-cache")
)

import matplotlib.pyplot as plt
import numpy as np


SIZES = (128, 256, 512, 1024, 2048, 4096, 8192)
HERE = Path(__file__).resolve().parent
DEFAULT_EXECUTABLE = HERE / "exec"
DEFAULT_OUTPUT = HERE / "plots" / "plot1.pdf"


def input_for_executable(values):
    return f"{len(values)}\n" + " ".join(f"{value:.17g}" for value in values) + "\n"


def benchmark_cpp_fft(executable, program_input, repeats, timeout):
    completed = subprocess.run(
        [str(executable), "--benchmark", str(repeats)],
        input=program_input,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{executable} a iesit cu codul {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    try:
        return float(completed.stdout.strip())
    except ValueError as error:
        raise RuntimeError(
            f"timp C++ invalid: {completed.stdout.strip()!r}"
        ) from error


def median_time(function, repeats, warmup=True):
    if warmup:
        function()
    measurements = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        measurements.append(time.perf_counter() - start)
    return statistics.median(measurements)


def direct_dft(values, block_size=128):
    """DFT O(N^2), calculata pe blocuri pentru a limita memoria folosita."""
    n = len(values)
    sample_indices = np.arange(n)
    result = np.empty(n, dtype=np.complex128)

    for begin in range(0, n, block_size):
        end = min(begin + block_size, n)
        frequency_indices = np.arange(begin, end)[:, np.newaxis]
        matrix_block = np.exp(
            -2j * np.pi * frequency_indices * sample_indices / n
        )
        result[begin:end] = matrix_block @ values
    return result


def verify_executable(executable, timeout):
    values = np.array([1.0, 2.0, 3.0, 4.0])
    completed = subprocess.run(
        [str(executable)],
        input=input_for_executable(values),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=True,
    )
    obtained = []
    for line in completed.stdout.splitlines():
        real, imaginary = line.strip()[1:-1].split(",")
        obtained.append(complex(float(real), float(imaginary)))
    if not np.allclose(obtained, np.fft.fft(values), rtol=2e-5, atol=2e-4):
        raise RuntimeError("executabilul nu coincide cu numpy.fft pentru testul de control")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exec", dest="executable", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cpp-repeats", type=int, default=7)
    parser.add_argument("--numpy-repeats", type=int, default=31)
    parser.add_argument("--dft-repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    executable = args.executable.resolve()
    if not executable.is_file():
        parser.error(f"executabilul nu exista: {executable}")
    if min(args.cpp_repeats, args.numpy_repeats, args.dft_repeats) < 1:
        parser.error("numarul de repetari trebuie sa fie pozitiv")

    verify_executable(executable, args.timeout)
    generator = np.random.default_rng(42)
    cpp_times = []
    dft_times = []
    numpy_times = []

    print(
        f"{'N':>6} {'DFT proprie [ms]':>18} {'FFT C++ [ms]':>19} "
        f"{'numpy.fft [ms]':>16}"
    )
    for n in SIZES:
        values = generator.uniform(-1.0, 1.0, n)
        program_input = input_for_executable(values)

        cpp_time = benchmark_cpp_fft(
            executable,
            program_input,
            args.cpp_repeats,
            args.timeout,
        )
        numpy_time = median_time(lambda: np.fft.fft(values), args.numpy_repeats)
        dft_time = median_time(
            lambda: direct_dft(values), args.dft_repeats, warmup=False
        )

        cpp_times.append(cpp_time)
        numpy_times.append(numpy_time)
        dft_times.append(dft_time)
        print(
            f"{n:6d} {dft_time * 1e3:18.3f} {cpp_time * 1e3:19.3f} "
            f"{numpy_time * 1e3:16.6f}"
        )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(SIZES, dft_times, "o-", label="DFT proprie, Python")
    axis.plot(SIZES, cpp_times, "o-", label="FFT proprie, C++")
    axis.plot(SIZES, numpy_times, "o-", label="numpy.fft.fft")
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks(SIZES)
    axis.set_xticklabels([str(n) for n in SIZES])
    axis.set_xlabel("Dimensiunea vectorului N")
    axis.set_ylabel("Timp median [s] - scara logaritmica")
    axis.set_title("Compararea timpilor de executie DFT si FFT")
    axis.grid(True, which="both", linestyle="--", alpha=0.45)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(figure)
    print(f"\nGrafic salvat in {output}")
    print("Timpul C++ este masurat intern doar in jurul apelului solver.fft().")


if __name__ == "__main__":
    main()
