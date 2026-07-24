from __future__ import annotations
from collections import Counter
from .circuit import Circuit
from .result import RunResult, StatevectorResult
from . import _ffi
import numpy as np

def _execute(circuit: Circuit) -> int:
    """Allocate a simulator handle, batch-apply all gates, return handle."""
    h = _ffi.create(circuit.n_qubits)
    _ffi.apply_gates(h, circuit.ops)
    return h

def run(circuit: Circuit, shots: int = 1024, backend: str = "mps") -> RunResult:
    h = _execute(circuit)
    raw = _ffi.sample(h, shots)
    _ffi.destroy(h)

    fmt = f"{{:0{circuit.n_qubits}b}}"
    counts = Counter(fmt.format(int(v)) for v in raw)
    return RunResult(counts=dict(counts), shots=shots, n_qubits=circuit.n_qubits)

def statevector(circuit: Circuit) -> StatevectorResult:
    h = _execute(circuit)
    sv = _ffi.get_statevector(h, circuit.n_qubits)
    _ffi.destroy(h)
    return StatevectorResult(vector=sv, n_qubits=circuit.n_qubits)

def expect(circuit: Circuit, hamiltonian) -> float:
    from .hamiltonian import _pack_hamiltonian
    import ctypes
    h = _execute(circuit)
    pauli_ids, qubits, coeffs = _pack_hamiltonian(hamiltonian)
    n_terms = len(coeffs)
    n_qpt   = circuit.n_qubits
    result  = _ffi._lib.mq_expect(
        h,
        pauli_ids.ctypes.data_as(ctypes.c_void_p),
        qubits.ctypes.data_as(ctypes.c_void_p),
        coeffs.ctypes.data_as(ctypes.c_void_p),
        n_terms, n_qpt,
    )
    _ffi.destroy(h)
    return float(result)
