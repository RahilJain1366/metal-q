from dataclasses import dataclass
from typing import Dict
import numpy as np

@dataclass
class RunResult:
    counts:   Dict[str, int]
    shots:    int
    n_qubits: int

    def most_probable(self) -> str:
        return max(self.counts, key=self.counts.get)

    def probabilities(self) -> Dict[str, float]:
        return {k: v / self.shots for k, v in self.counts.items()}

    def __repr__(self):
        top = sorted(self.counts.items(), key=lambda x: -x[1])[:5]
        return f"RunResult(shots={self.shots}, top={top})"

@dataclass
class StatevectorResult:
    vector:   np.ndarray   # complex128
    n_qubits: int

    def probabilities(self) -> np.ndarray:
        return np.abs(self.vector) ** 2

    def __repr__(self):
        return f"StatevectorResult(n_qubits={self.n_qubits}, dim={len(self.vector)})"