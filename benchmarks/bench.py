"""
bench.py — Wall-clock comparison of MetalQ, Qiskit Aer, and PennyLane.

Circuits tested
  • Random-layer circuit  : alternating layers of random single-qubit
                            rotations and CNOT ladders (depth = 2*layers)
  • QFT                   : quantum Fourier transform on all qubits

Metrics
  • statevector wall-clock time (median of REPS runs, in milliseconds)

Usage
  python benchmarks/bench.py
"""

import time
import statistics
import sys
import numpy as np

QUBIT_COUNTS = [4, 8, 12, 16, 20]
REPS = 5          # median over this many runs
LAYERS = 4        # entangling layers per random circuit

# ── helpers ──────────────────────────────────────────────────────────────────

def _random_angles(n_params: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 2 * np.pi, n_params).tolist()

def _median_ms(fn, reps: int) -> float:
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


# ── MetalQ ───────────────────────────────────────────────────────────────────

def metalq_random(n: int) -> float:
    from metalq import Circuit, statevector
    angles = _random_angles(n * LAYERS * 3)
    idx = 0
    def run():
        qc = Circuit(n)
        nonlocal idx; idx = 0
        for _ in range(LAYERS):
            for q in range(n):
                qc.rx(angles[idx], q); idx += 1
                qc.ry(angles[idx], q); idx += 1
                qc.rz(angles[idx], q); idx += 1
            for q in range(n - 1):
                qc.cx(q, q + 1)
        statevector(qc)
    return _median_ms(run, REPS)

def metalq_qft(n: int) -> float:
    from metalq import Circuit, statevector
    import math
    def run():
        qc = Circuit(n)
        qubits = list(range(n))
        # inline QFT to avoid cp overhead counting
        for j in range(n):
            qc.h(qubits[j])
            for k in range(j + 1, n):
                angle = math.pi / (2 ** (k - j))
                qc.cp(angle, qubits[k], qubits[j])
        for i in range(n // 2):
            qc.swap(qubits[i], qubits[n - 1 - i])
        statevector(qc)
    return _median_ms(run, REPS)


# ── Qiskit Aer ───────────────────────────────────────────────────────────────

def aer_random(n: int) -> float:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit.compiler import transpile
    sim = AerSimulator(method="statevector")
    angles = _random_angles(n * LAYERS * 3)
    def run():
        qc = QuantumCircuit(n)
        idx = 0
        for _ in range(LAYERS):
            for q in range(n):
                qc.rx(angles[idx], q); idx += 1
                qc.ry(angles[idx], q); idx += 1
                qc.rz(angles[idx], q); idx += 1
            for q in range(n - 1):
                qc.cx(q, q + 1)
        qc.save_statevector()
        t = transpile(qc, sim)
        sim.run(t).result()
    return _median_ms(run, REPS)

def aer_qft(n: int) -> float:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import QFT
    from qiskit_aer import AerSimulator
    from qiskit.compiler import transpile
    sim = AerSimulator(method="statevector")
    def run():
        qc = QuantumCircuit(n)
        qc.compose(QFT(n), inplace=True)
        qc.save_statevector()
        t = transpile(qc, sim)
        sim.run(t).result()
    return _median_ms(run, REPS)


# ── PennyLane ────────────────────────────────────────────────────────────────

def pennylane_random(n: int) -> float:
    import pennylane as qml
    dev = qml.device("default.qubit", wires=n)
    angles = _random_angles(n * LAYERS * 3)

    @qml.qnode(dev)
    def circuit():
        idx = 0
        for _ in range(LAYERS):
            for q in range(n):
                qml.RX(angles[idx], wires=q); idx += 1
                qml.RY(angles[idx], wires=q); idx += 1
                qml.RZ(angles[idx], wires=q); idx += 1
            for q in range(n - 1):
                qml.CNOT(wires=[q, q + 1])
        return qml.state()

    return _median_ms(circuit, REPS)

def pennylane_qft(n: int) -> float:
    import pennylane as qml
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circuit():
        qml.QFT(wires=range(n))
        return qml.state()

    return _median_ms(circuit, REPS)


# ── table formatter ───────────────────────────────────────────────────────────

def fmt(ms: float | None) -> str:
    if ms is None:
        return "  OOM  "
    if ms < 1:
        return f"{ms*1000:.1f} µs"
    if ms < 1000:
        return f"{ms:.1f} ms"
    return f"{ms/1000:.2f} s"

def run_suite(label: str, metalq_fn, aer_fn, pl_fn):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    header = f"{'Qubits':>7} | {'MetalQ':>10} | {'Qiskit Aer':>10} | {'PennyLane':>10} | {'Aer speedup':>11}"
    print(header)
    print("-" * (len(header)+1))
    results = []
    for n in QUBIT_COUNTS:
        mq = metalq_fn(n)
        try:
            ae = aer_fn(n)
        except Exception:
            ae = None
        try:
            pl = pl_fn(n)
        except Exception:
            pl = None
        ratio = f"{ae/mq:.1f}x" if ae is not None else "—"
        print(f"{n:>7} | {fmt(mq):>10} | {fmt(ae) if ae is not None else 'error':>10} | {fmt(pl) if pl is not None else 'error':>10} | {ratio:>11}")
        sys.stdout.flush()
        results.append((n, mq, ae, pl))
    return results


if __name__ == "__main__":
    print(f"MetalQ Benchmark  (median of {REPS} runs, {LAYERS} entangling layers)")
    print(f"Platform: Apple Silicon GPU (Metal) vs CPU simulators")
    print(f"Qubit counts: {QUBIT_COUNTS}")

    r1 = run_suite("Random Rotation+CNOT Circuit", metalq_random, aer_random, pennylane_random)
    r2 = run_suite("QFT Circuit", metalq_qft, aer_qft, pennylane_qft)

    print("\n[done]")
