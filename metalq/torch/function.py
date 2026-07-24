import torch
import ctypes
from torch.autograd import Function
from ..runner import expect
from .. import _ffi
import numpy as np


class QuantumFunction(Function):
    @staticmethod
    def forward(ctx, thetas: torch.Tensor, circuit, hamiltonian, param_names):
        ctx.save_for_backward(thetas)
        ctx.circuit     = circuit
        ctx.hamiltonian = hamiltonian
        ctx.param_names = param_names

        bound = circuit.bind_parameters(dict(zip(param_names, thetas.detach().tolist())))
        val = expect(bound, hamiltonian)
        return torch.tensor(val, dtype=torch.float64)

    @staticmethod
    def backward(ctx, grad_output):
        thetas, = ctx.saved_tensors
        grads = _adjoint_grad(
            ctx.circuit, ctx.hamiltonian,
            ctx.param_names, thetas.detach().tolist())
        return (grad_output * torch.tensor(grads, dtype=torch.float64),
                None, None, None)


def _adjoint_grad(circuit, hamiltonian, param_names, theta_values):
    """Calls mq_adjoint_gradient via FFI and returns gradient list."""
    from ..hamiltonian import _pack_hamiltonian
    bound = circuit.bind_parameters(dict(zip(param_names, theta_values)))
    pauli_ids, qubits, coeffs = _pack_hamiltonian(hamiltonian)

    n_gates = len(bound.ops)
    GateArr    = (_ffi.GateParamStruct * n_gates)()
    param_mask = (ctypes.c_uint8 * n_gates)()
    grad_out   = (ctypes.c_double * n_gates)()

    for i, op in enumerate(bound.ops):
        GateArr[i].gate_id = op.gate_id
        GateArr[i].target  = op.target
        GateArr[i].control = op.control
        GateArr[i].theta   = float(op.resolved_theta())
        GateArr[i].phi     = float(op.phi)
        GateArr[i].lam     = float(op.lam)
        param_mask[i] = 1 if circuit.ops[i].is_parameterized() else 0

    h = _ffi.create(circuit.n_qubits)
    _ffi._lib.mq_adjoint_gradient(
        h,
        pauli_ids.ctypes.data_as(ctypes.c_void_p),
        qubits.ctypes.data_as(ctypes.c_void_p),
        coeffs.ctypes.data_as(ctypes.c_void_p),
        len(coeffs), circuit.n_qubits,
        GateArr, n_gates, param_mask, grad_out,
    )
    _ffi.destroy(h)
    return [grad_out[i] for i in range(n_gates) if param_mask[i]]
