from __future__ import annotations
from typing import List, Optional
from .gate import GateOp, Parameter
import math

class Circuit:
    def __init__(self, n_qubits: int):
        self.n_qubits  = n_qubits
        self.ops: List[GateOp] = []

    # ── single-qubit gates ──────────────────────────────────────────────────

    def h(self, qubit: int):
        self.ops.append(GateOp("H", qubit)); return self

    def x(self, qubit: int):
        self.ops.append(GateOp("X", qubit)); return self

    def y(self, qubit: int):
        self.ops.append(GateOp("Y", qubit)); return self

    def z(self, qubit: int):
        self.ops.append(GateOp("Z", qubit)); return self

    def t(self, qubit: int):
        self.ops.append(GateOp("T", qubit)); return self

    def s(self, qubit: int):
        self.ops.append(GateOp("S", qubit)); return self

    def rx(self, theta, qubit: int):
        self.ops.append(GateOp("RX", qubit, theta=theta)); return self

    def ry(self, theta, qubit: int):
        self.ops.append(GateOp("RY", qubit, theta=theta)); return self

    def rz(self, theta, qubit: int):
        self.ops.append(GateOp("RZ", qubit, theta=theta)); return self

    def u(self, theta, phi, lam, qubit: int):
        self.ops.append(GateOp("U", qubit, theta=theta, phi=phi, lam=lam))
        return self

    # ── two-qubit gates ─────────────────────────────────────────────────────

    def cx(self, control: int, target: int):
        self.ops.append(GateOp("CNOT", target, control=control)); return self

    def cz(self, control: int, target: int):
        self.ops.append(GateOp("CZ", target, control=control)); return self

    def swap(self, q0: int, q1: int):
        self.ops.append(GateOp("SWAP", q1, control=q0)); return self

    # ── introspection ────────────────────────────────────────────────────────

    def parameters(self) -> List[Parameter]:
        return [op.theta for op in self.ops if op.is_parameterized()]

    def bind_parameters(self, values: dict) -> "Circuit":
        """Returns a new Circuit with Parameters resolved to floats."""
        c = Circuit(self.n_qubits)
        for op in self.ops:
            if op.is_parameterized() and op.theta.name in values:
                import copy
                new_op = copy.copy(op)
                new_op.theta = values[op.theta.name]
                c.ops.append(new_op)
            else:
                c.ops.append(op)
        return c

    def depth(self) -> int:
        # Simple layer counting — greedy left-to-right
        latest = [-1] * self.n_qubits
        layers = 0
        for op in self.ops:
            qubits = [op.target]
            if op.control >= 0: qubits.append(op.control)
            layer = max(latest[q] for q in qubits) + 1
            for q in qubits: latest[q] = layer
            layers = max(layers, layer + 1)
        return layers

    def __repr__(self):
        return (f"Circuit(n_qubits={self.n_qubits}, "
                f"depth={self.depth()}, gates={len(self.ops)})")