from __future__ import annotations
from typing import List
from .gate import GateOp, Parameter
import math


class Circuit:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.ops: List[GateOp] = []

    # ── Single-qubit gates ────────────────────────────────────────────────────

    def h(self, qubit: int) -> "Circuit":
        self.ops.append(GateOp("H", qubit)); return self

    def x(self, qubit: int) -> "Circuit":
        self.ops.append(GateOp("X", qubit)); return self

    def y(self, qubit: int) -> "Circuit":
        self.ops.append(GateOp("Y", qubit)); return self

    def z(self, qubit: int) -> "Circuit":
        self.ops.append(GateOp("Z", qubit)); return self

    def t(self, qubit: int) -> "Circuit":
        self.ops.append(GateOp("T", qubit)); return self

    def s(self, qubit: int) -> "Circuit":
        self.ops.append(GateOp("S", qubit)); return self

    def tdg(self, qubit: int) -> "Circuit":
        """T† = [[1,0],[0,e^{-iπ/4}]] — implemented via U(0,0,-π/4)."""
        return self.u(0.0, 0.0, -math.pi / 4, qubit)

    def sdg(self, qubit: int) -> "Circuit":
        """S† = [[1,0],[0,-i]] — implemented via U(0,0,-π/2)."""
        return self.u(0.0, 0.0, -math.pi / 2, qubit)

    def rx(self, theta, qubit: int) -> "Circuit":
        self.ops.append(GateOp("RX", qubit, theta=theta)); return self

    def ry(self, theta, qubit: int) -> "Circuit":
        self.ops.append(GateOp("RY", qubit, theta=theta)); return self

    def rz(self, theta, qubit: int) -> "Circuit":
        self.ops.append(GateOp("RZ", qubit, theta=theta)); return self

    def u(self, theta, phi, lam, qubit: int) -> "Circuit":
        self.ops.append(GateOp("U", qubit, theta=theta, phi=phi, lam=lam))
        return self

    # ── Two-qubit gates ───────────────────────────────────────────────────────

    def cx(self, control: int, target: int) -> "Circuit":
        self.ops.append(GateOp("CNOT", target, control=control)); return self

    def cz(self, control: int, target: int) -> "Circuit":
        self.ops.append(GateOp("CZ", target, control=control)); return self

    def swap(self, q0: int, q1: int) -> "Circuit":
        self.ops.append(GateOp("SWAP", q1, control=q0)); return self

    # ── Controlled single-qubit gates ─────────────────────────────────────────
    # The Metal kernel already has the control guard for every single-qubit
    # gate — these just expose that path via the Python API.

    def c(self, gate: str, control: int, target: int,
          theta=0.0, phi=0.0, lam=0.0) -> "Circuit":
        """Generic controlled single-qubit gate."""
        self.ops.append(GateOp(gate.upper(), target, control=control,
                               theta=theta, phi=phi, lam=lam))
        return self

    def crx(self, theta, control: int, target: int) -> "Circuit":
        return self.c("RX", control, target, theta=theta)

    def cry(self, theta, control: int, target: int) -> "Circuit":
        return self.c("RY", control, target, theta=theta)

    def crz(self, theta, control: int, target: int) -> "Circuit":
        return self.c("RZ", control, target, theta=theta)

    def cp(self, lam, control: int, target: int) -> "Circuit":
        """Controlled-phase CP(λ) = diag(1,1,1,e^{iλ}).

        Route A: CRZ(λ) · RZ(λ/2)_control  = e^{-iλ/4} · CP(λ).
        The global phase factor e^{-iλ/4} is unobservable on an
        uncontrolled gate and is physically correct for QFT use.
        """
        self.crz(lam, control, target)
        self.rz(lam / 2, control)
        return self

    # ── Toffoli (CCX) ─────────────────────────────────────────────────────────

    def ccx(self, c1: int, c2: int, target: int) -> "Circuit":
        """Toffoli gate — standard 6-CNOT decomposition (Nielsen & Chuang)."""
        self.h(target)
        self.cx(c1, target)
        self.tdg(target)
        self.cx(c2, target)
        self.t(target)
        self.cx(c1, target)
        self.tdg(target)
        self.cx(c2, target)
        self.t(c2)
        self.t(target)
        self.h(target)
        self.cx(c1, c2)
        self.t(c1)
        self.tdg(c2)
        self.cx(c1, c2)
        return self

    # ── Multi-controlled gates ────────────────────────────────────────────────

    def mcx(self, controls: List[int], target: int,
            ancillas: List[int]) -> "Circuit":
        """Multi-controlled X using a Toffoli AND-ladder.

        Requires len(ancillas) >= max(0, len(controls) - 2).
        Ancillas must start in |0⟩ and are returned to |0⟩ (uncomputed).
        """
        k = len(controls)
        if k == 0:
            self.x(target)
        elif k == 1:
            self.cx(controls[0], target)
        elif k == 2:
            self.ccx(controls[0], controls[1], target)
        else:
            n_anc = k - 2
            assert len(ancillas) >= n_anc, (
                f"MCX({k} controls) needs {n_anc} ancillas, got {len(ancillas)}")
            # Forward AND-ladder
            self.ccx(controls[0], controls[1], ancillas[0])
            for i in range(1, n_anc):
                self.ccx(ancillas[i - 1], controls[i + 1], ancillas[i])
            self.ccx(ancillas[n_anc - 1], controls[k - 1], target)
            # Uncompute
            for i in range(n_anc - 1, 0, -1):
                self.ccx(ancillas[i - 1], controls[i + 1], ancillas[i])
            self.ccx(controls[0], controls[1], ancillas[0])
        return self

    def mcry(self, theta, controls: List[int], target: int,
             ancillas: List[int]) -> "Circuit":
        """Multi-controlled RY.

        Computes AND of all controls into ancillas[-1], applies CRY,
        then uncomputes.  Requires len(ancillas) >= max(1, len(controls)-1).
        """
        k = len(controls)
        if k == 0:
            self.ry(theta, target)
        elif k == 1:
            self.cry(theta, controls[0], target)
        else:
            n_needed = k - 1
            assert len(ancillas) >= n_needed, (
                f"MCRY({k} controls) needs {n_needed} ancillas, got {len(ancillas)}")
            and_anc    = ancillas[-1]
            ladder_anc = list(ancillas[:-1])
            # Compute AND of controls → and_anc
            self.mcx(controls, and_anc, ladder_anc)
            # Apply CRY
            self.cry(theta, and_anc, target)
            # Uncompute
            self.mcx(controls, and_anc, ladder_anc)
        return self

    # ── QFT / IQFT ────────────────────────────────────────────────────────────

    def qft(self, qubits: List[int], swaps: bool = True) -> "Circuit":
        """Quantum Fourier Transform on the given qubit indices.

        Convention: qubit 0 of the list is the most-significant bit of
        the output register.  The final SWAP layer bit-reverses to match
        the standard |k⟩ → (1/√N) Σ_j e^{2πijk/N} |j⟩ definition.
        """
        n = len(qubits)
        for j in range(n):
            self.h(qubits[j])
            for k in range(j + 1, n):
                angle = math.pi / (2 ** (k - j))
                self.cp(angle, qubits[k], qubits[j])
        if swaps:
            for i in range(n // 2):
                self.swap(qubits[i], qubits[n - 1 - i])
        return self

    def iqft(self, qubits: List[int], swaps: bool = True) -> "Circuit":
        """Inverse Quantum Fourier Transform."""
        n = len(qubits)
        if swaps:
            for i in range(n // 2):
                self.swap(qubits[i], qubits[n - 1 - i])
        for j in range(n - 1, -1, -1):
            for k in range(n - 1, j, -1):
                angle = -math.pi / (2 ** (k - j))
                self.cp(angle, qubits[k], qubits[j])
            self.h(qubits[j])
        return self

    # ── Introspection ─────────────────────────────────────────────────────────

    def parameters(self) -> List[Parameter]:
        return [op.theta for op in self.ops if op.is_parameterized()]

    def bind_parameters(self, values: dict) -> "Circuit":
        """Return a new Circuit with Parameters resolved to floats."""
        import copy
        c = Circuit(self.n_qubits)
        for op in self.ops:
            if op.is_parameterized() and op.theta.name in values:
                new_op = copy.copy(op)
                new_op.theta = values[op.theta.name]
                c.ops.append(new_op)
            else:
                c.ops.append(op)
        return c

    def depth(self) -> int:
        latest = [-1] * self.n_qubits
        layers = 0
        for op in self.ops:
            qubits = [op.target]
            if op.control >= 0:
                qubits.append(op.control)
            layer = max(latest[q] for q in qubits) + 1
            for q in qubits:
                latest[q] = layer
            layers = max(layers, layer + 1)
        return layers

    def gate_counts(self) -> dict:
        """Return a dict of gate_name → count, plus totals."""
        from collections import Counter
        c = Counter(op.name for op in self.ops)
        c["total"] = len(self.ops)
        c["two_qubit"] = sum(v for k, v in c.items()
                             if k in ("CNOT", "CZ", "SWAP") and k != "total")
        return dict(c)

    def __repr__(self):
        return (f"Circuit(n_qubits={self.n_qubits}, "
                f"depth={self.depth()}, gates={len(self.ops)})")
