#!/usr/bin/env python3
"""Teste black-box pentru executabilul FFT.

Executabilul primeste de la stdin:
    n
    x0 x1 ... x(n-1)

si trebuie sa afiseze cate un complex `(real,imag)` pe linie. Pentru n care
nu este putere a lui 2, referinta completeaza vectorul cu zerouri pana la
urmatoarea putere a lui 2.

Rulare din radacina proiectului:
    ./.venv/bin/python labs/lab4/ex1/test_fft.py
    ./.venv/bin/python labs/lab4/ex1/test_fft.py correctness
    ./.venv/bin/python labs/lab4/ex1/test_fft.py timing
    ./.venv/bin/python labs/lab4/ex1/test_fft.py --exec labs/lab4/ex1/exec
"""

import argparse
import cmath
import math
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import time


DEFAULT_EXECUTABLE = Path(__file__).resolve().parent / "exec"
BENCHMARK_SIZES = (128, 256, 512, 1024, 2048, 4096, 8192)
COMPLEXITY_SIZES = (8192, 16384, 32768, 65536, 131072)
COMPLEX_LINE = re.compile(
    r"\s*\(\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"\s*,\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"\s*\)\s*"
)


def pad_to_power_of_two(values):
    if not values:
        return []
    size = 1 << (len(values) - 1).bit_length()
    return values + [0.0] * (size - len(values))


def direct_dft(values, sign):
    """Referinta O(N^2); sign=-1 pentru DFT si sign=+1 pentru conventia opusa."""
    n = len(values)
    return [
        sum(
            value * cmath.exp(sign * 2j * math.pi * frequency * time / n)
            for time, value in enumerate(values)
        )
        for frequency in range(n)
    ]


def parse_output(stdout):
    result = []
    for line_number, line in enumerate(stdout.splitlines(), start=1):
        match = COMPLEX_LINE.fullmatch(line)
        if match is None:
            raise AssertionError(
                f"linia {line_number} nu are formatul (real,imag): {line!r}"
            )
        result.append(complex(float(match.group(1)), float(match.group(2))))
    return result


def run_program(executable, values, timeout):
    program_input = f"{len(values)}\n"
    if values:
        program_input += " ".join(f"{value:.17g}" for value in values) + "\n"

    completed = subprocess.run(
        [str(executable)],
        input=program_input,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        detail = f": {stderr}" if stderr else ""
        raise AssertionError(f"executabilul a iesit cu codul {completed.returncode}{detail}")
    if completed.stderr.strip():
        raise AssertionError(f"executabilul a scris in stderr: {completed.stderr.strip()!r}")
    return parse_output(completed.stdout)


def check_case(executable, values, sign, timeout, abs_tolerance, rel_tolerance):
    padded = pad_to_power_of_two(values)
    expected = direct_dft(padded, sign) if padded else []
    actual = run_program(executable, values, timeout)

    if len(actual) != len(expected):
        raise AssertionError(
            f"a afisat {len(actual)} coeficienti, asteptat {len(expected)} "
            f"(n={len(values)}, dupa padding N={len(padded)})"
        )

    for index, (obtained, wanted) in enumerate(zip(actual, expected)):
        if not cmath.isclose(
            obtained, wanted, abs_tol=abs_tolerance, rel_tol=rel_tolerance
        ):
            error = abs(obtained - wanted)
            raise AssertionError(
                f"X[{index}]={obtained:.9g}, asteptat {wanted:.9g}, "
                f"eroare absoluta={error:.3g}"
            )


def test_cases():
    cases = [
        ("vector gol", []),
        ("un element", [3.5]),
        ("doi elementi", [1.0, -2.0]),
        ("zerouri", [0.0] * 8),
        ("impuls in origine", [1.0] + [0.0] * 7),
        ("impuls deplasat", [0.0, 1.0] + [0.0] * 6),
        ("semnal constant", [2.0] * 8),
        ("semnal alternant", [1.0, -1.0] * 4),
        ("exemplul 1..4", [1.0, 2.0, 3.0, 4.0]),
        ("exemplul 1..8", [float(x) for x in range(1, 9)]),
        ("padding n=3", [1.0, 2.0, 3.0]),
        ("padding n=5", [1.0, -2.0, 0.5, 4.0, -1.0]),
        ("fractiuni si negative", [-2.5, 0.125, 3.75, -1.0]),
        (
            "cosinus, frecventa 3",
            [math.cos(2 * math.pi * 3 * t / 16) for t in range(16)],
        ),
        (
            "sinus, frecventa 3",
            [math.sin(2 * math.pi * 3 * t / 16) for t in range(16)],
        ),
    ]

    generator = random.Random(42)
    for n in (3, 4, 7, 8, 15, 16, 31, 32, 63, 64):
        values = [generator.uniform(-10.0, 10.0) for _ in range(n)]
        cases.append((f"aleator n={n}", values))
    return cases


def program_input(values):
    return f"{len(values)}\n" + " ".join(
        f"{value:.17g}" for value in values
    ) + "\n"


def median_cpp_time(executable, values, repeats, timeout):
    data = program_input(values)
    completed = subprocess.run(
        [str(executable), "--benchmark", str(repeats)],
        input=data,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"executabilul a iesit cu codul {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    try:
        return float(completed.stdout.strip())
    except ValueError as error:
        raise AssertionError(
            f"executabilul a afisat un timp invalid: {completed.stdout.strip()!r}"
        ) from error


def median_numpy_time(np, values, repeats):
    array = np.asarray(values)
    np.fft.fft(array)  # incalzire
    measurements = []
    for _ in range(repeats):
        start = time.perf_counter()
        np.fft.fft(array)
        measurements.append(time.perf_counter() - start)
    return statistics.median(measurements)


def logarithmic_slope(sizes, times):
    x = [math.log(size) for size in sizes]
    y = [math.log(measurement) for measurement in times]
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    return sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / sum(
        (a - x_mean) ** 2 for a in x
    )


def run_timing_tests(executable, timeout, cpp_repeats, numpy_repeats):
    try:
        import numpy as np
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "numpy nu este instalat in acest interpreter; ruleaza cu "
            "./.venv/bin/python labs/lab4/ex1/test_fft.py"
        ) from error

    generator = random.Random(2026)
    cpp_times = []
    numpy_times = []

    print("\nTimpi FFT: executabil C++ fata de numpy.fft.fft")
    print(
        f"{'N':>6} {'FFT C++ [ms]':>16} {'NumPy [ms]':>13} "
        f"{'C++ / (N log2 N) [ns]':>26} {'raport C++ la N/2':>20}"
    )
    previous = None
    for n in BENCHMARK_SIZES:
        values = [generator.uniform(-1.0, 1.0) for _ in range(n)]
        cpp_time = median_cpp_time(executable, values, cpp_repeats, timeout)
        numpy_time = median_numpy_time(np, values, numpy_repeats)
        cpp_times.append(cpp_time)
        numpy_times.append(numpy_time)

        ratio = f"{cpp_time / previous:.3f}" if previous is not None else "-"
        normalized = cpp_time * 1e9 / (n * math.log2(n))
        print(
            f"{n:6d} {cpp_time * 1e3:16.3f} {numpy_time * 1e3:13.6f} "
            f"{normalized:26.3f} {ratio:>20}"
        )
        previous = cpp_time

    # Pentru testul de ordin folosim N mai mari. 8192 a fost deja masurat mai sus.
    complexity_times = [cpp_times[-1]]
    complexity_repeats = min(cpp_repeats, 5)
    print("\nMasuratori suplimentare pentru ordinul de complexitate:")
    print(f"{'N':>8} {'FFT C++ [ms]':>16} {'t/(N log2 N) [ns]':>22}")
    print(
        f"{COMPLEXITY_SIZES[0]:8d} {complexity_times[0] * 1e3:16.3f} "
        f"{complexity_times[0] * 1e9 / (COMPLEXITY_SIZES[0] * math.log2(COMPLEXITY_SIZES[0])):22.3f}"
    )
    for n in COMPLEXITY_SIZES[1:]:
        values = [generator.uniform(-1.0, 1.0) for _ in range(n)]
        measurement = median_cpp_time(
            executable, values, complexity_repeats, timeout
        )
        complexity_times.append(measurement)
        print(
            f"{n:8d} {measurement * 1e3:16.3f} "
            f"{measurement * 1e9 / (n * math.log2(n)):22.3f}"
        )

    large_sizes = COMPLEXITY_SIZES[-4:]
    large_times = complexity_times[-4:]
    slope = logarithmic_slope(large_sizes, large_times)
    normalized = [
        measurement / (n * math.log2(n))
        for n, measurement in zip(large_sizes, large_times)
    ]
    normalized_spread = max(normalized) / min(normalized)

    # Pentru N log N panta efectiva este putin peste 1; pentru N^2 tinde la 2.
    slope_ok = 0.70 <= slope <= 1.65
    spread_ok = normalized_spread <= 4.0
    print("\nTeste empirice de complexitate pentru ultimele 4 dimensiuni:")
    print(
        f"  {'OK' if slope_ok else 'FAIL'}  panta log(t)/log(N) = {slope:.3f} "
        "(interval acceptat 0.70..1.65)"
    )
    print(
        f"  {'OK' if spread_ok else 'FAIL'}  variatia t/(N log2 N) = "
        f"{normalized_spread:.3f}x (maximum acceptat 4x)"
    )
    print(
        "Timpul C++ este masurat intern doar in jurul solver.fft(). "
        "Testele sunt dovezi empirice, nu o demonstratie matematica."
    )
    return int(not (slope_ok and spread_ok))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("all", "correctness", "timing"),
        default="all",
    )
    parser.add_argument(
        "--exec",
        dest="executable",
        type=Path,
        default=DEFAULT_EXECUTABLE,
        help=f"executabilul testat (implicit: {DEFAULT_EXECUTABLE})",
    )
    parser.add_argument(
        "--sign",
        choices=("negative", "positive"),
        default="negative",
        help="semnul exponentului din definitia DFT (implicit: negative)",
    )
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument(
        "--abs-tol",
        type=float,
        default=2e-4,
        help="toleranta absoluta; cout foloseste implicit 6 cifre semnificative",
    )
    parser.add_argument("--rel-tol", type=float, default=2e-5)
    parser.add_argument("--cpp-repeats", type=int, default=7)
    parser.add_argument("--numpy-repeats", type=int, default=31)
    args = parser.parse_args()

    executable = args.executable.resolve()
    if not executable.is_file():
        parser.error(f"executabilul nu exista: {executable}")
    if args.timeout <= 0 or args.abs_tol < 0 or args.rel_tol < 0:
        parser.error("timeout trebuie sa fie pozitiv, iar tolerantele nenegative")
    if args.cpp_repeats < 1 or args.numpy_repeats < 1:
        parser.error("numarul de repetari trebuie sa fie pozitiv")

    sign = -1 if args.sign == "negative" else 1
    convention = "exp(-2*pi*i*k*t/N)" if sign == -1 else "exp(+2*pi*i*k*t/N)"
    failures = 0

    if args.mode in ("all", "correctness"):
        cases = test_cases()
        print(f"Executabil: {executable}")
        print(f"Referinta: DFT directa, {convention}")
        print("Padding: zerouri pana la urmatoarea putere a lui 2\n")

        correctness_failures = 0
        for name, values in cases:
            try:
                check_case(
                    executable,
                    values,
                    sign,
                    args.timeout,
                    args.abs_tol,
                    args.rel_tol,
                )
                print(f"  OK    {name}")
            except subprocess.TimeoutExpired:
                correctness_failures += 1
                print(f"  FAIL  {name}: executia a depasit {args.timeout:g} secunde")
            except (AssertionError, OSError, ValueError) as error:
                correctness_failures += 1
                print(f"  FAIL  {name}: {error}")

        passed = len(cases) - correctness_failures
        print(f"\nCorectitudine: {passed}/{len(cases)} teste trecute.")
        failures += correctness_failures

    if args.mode in ("all", "timing"):
        try:
            failures += run_timing_tests(
                executable,
                args.timeout,
                args.cpp_repeats,
                args.numpy_repeats,
            )
        except (AssertionError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            failures += 1
            print(f"\nFAIL benchmark: {error}")

    return int(failures != 0)


if __name__ == "__main__":
    sys.exit(main())
