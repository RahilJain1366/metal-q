from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

PAULI_ID = {"I": 0, "X": 1, "Y": 2, "Z": 3}

@dataclass
class PauliTerm:
    coeff:   complex
    ops:     List[Tuple[str, int]]   # [("Z", 0), ("Z", 1)]

    def __mul__(self, other: "PauliTerm") -> "PauliTerm":
        return PauliTerm(self.coeff * other.coeff,
                         self.ops + other.ops)

@dataclass
class Hamiltonian:
    terms: List[PauliTerm] = field(default_factory=list)

    def __add__(self, other: "Hamiltonian") -> "Hamiltonian":
        return Hamiltonian(self.terms + other.terms)

    def __mul__(self, scalar: float) -> "Hamiltonian":
        return Hamiltonian(
            [PauliTerm(t.coeff * scalar, t.ops) for t in self.terms])

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

def _single(pauli: str, qubit: int) -> Hamiltonian:
    return Hamiltonian([PauliTerm(1.0, [(pauli, qubit)])])

def Z(qubit: int) -> Hamiltonian: return _single("Z", qubit)
def X(qubit: int) -> Hamiltonian: return _single("X", qubit)
def Y(qubit: int) -> Hamiltonian: return _single("Y", qubit)
def I(qubit: int) -> Hamiltonian: return _single("I", qubit)

def _pack_hamiltonian(h: Hamiltonian):
    """Returns (pauli_ids, qubits, coeffs) as flat numpy arrays."""
    n_terms = len(h.terms)
    max_ops = max(len(t.ops) for t in h.terms)
    pauli_ids = np.zeros((n_terms, max_ops), dtype=np.uint8)
    qubits    = np.zeros((n_terms, max_ops), dtype=np.uint32)
    coeffs    = np.zeros(n_terms,            dtype=np.float64)
    for i, term in enumerate(h.terms):
        coeffs[i] = term.coeff.real
        for j, (p, q) in enumerate(term.ops):
            pauli_ids[i, j] = PAULI_ID[p]
            qubits[i, j]    = q
    return pauli_ids, qubits, coeffs