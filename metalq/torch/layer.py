from __future__ import annotations
import torch
import torch.nn as nn
from ..circuit import Circuit
from ..hamiltonian import Hamiltonian
from .function import QuantumFunction

class QuantumLayer(nn.Module):
    """A trainable quantum circuit as a PyTorch layer.

    Parameters map directly onto the circuit's Parameter objects,
    in the order circuit.parameters() returns them.
    """

    def __init__(self, circuit: Circuit, hamiltonian: Hamiltonian,
                 backend_name: str = "mps"):
        super().__init__()
        self.circuit      = circuit
        self.hamiltonian  = hamiltonian
        self.backend_name = backend_name

        # One nn.Parameter per circuit Parameter, initialized to 0
        param_names = [p.name for p in circuit.parameters()]
        self._param_names = param_names
        self.thetas = nn.Parameter(torch.zeros(len(param_names)))

    def forward(self, x=None) -> torch.Tensor:
        return QuantumFunction.apply(
            self.thetas,
            self.circuit,
            self.hamiltonian,
            self._param_names,
        )