from __future__ import annotations
import sympy
from dataclasses import dataclass, field
from typing import Optional

# Gate IDs must match gates.rs constants
GATE_IDS = dict(H=0, X=1, Y=2, Z=3, RX=4, RY=5, RZ=6,
                U=7, CNOT=8, CZ=9, SWAP=10, T=11, S=12)

class Parameter:
    """A symbolic trainable angle."""
    def __init__(self, name: str):
        self.name   = name
        self.symbol = sympy.Symbol(name)
        self._value: Optional[float] = None

    def bind(self, value: float) -> Parameter:
        self._value = float(value)
        return self

    @property
    def value(self) -> float:
        if self._value is None:
            raise ValueError(f"Parameter '{self.name}' has no bound value")
        return self._value

    def __repr__(self):
        return f"Parameter('{self.name}', value={self._value})"

@dataclass
class GateOp:
    name:    str
    target:  int
    control: int        = -1
    theta:   object     = 0.0   # float or Parameter
    phi:     float      = 0.0
    lam:     float      = 0.0

    @property
    def gate_id(self) -> int:
        return GATE_IDS[self.name]

    def resolved_theta(self) -> float:
        return self.theta.value if isinstance(self.theta, Parameter) else float(self.theta)

    def is_parameterized(self) -> bool:
        return isinstance(self.theta, Parameter)