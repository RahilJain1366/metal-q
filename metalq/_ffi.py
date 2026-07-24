import ctypes, os, pathlib
import numpy as np

def _load_lib():
    candidates = [
        pathlib.Path(__file__).parent.parent / "target/release/libmetalq_native.dylib",
        pathlib.Path(__file__).parent / "libmetalq_native.dylib",
    ]
    for p in candidates:
        if p.exists():
            return ctypes.CDLL(str(p))
    raise FileNotFoundError("libmetalq_native.dylib not found. Run `cargo build --release`.")

_lib = _load_lib()

# ── GateParam struct (must match #[repr(C)] GateParam in lib.rs) ─────────────

class GateParamStruct(ctypes.Structure):
    _fields_ = [
        ("gate_id", ctypes.c_uint32),
        ("target",  ctypes.c_uint32),
        ("control", ctypes.c_int32),
        ("theta",   ctypes.c_float),
        ("phi",     ctypes.c_float),
        ("lam",     ctypes.c_float),
    ]

# ── Function signatures ───────────────────────────────────────────────────────

_lib.mq_create.restype  = ctypes.c_void_p
_lib.mq_create.argtypes = [ctypes.c_uint32]

_lib.mq_destroy.restype  = None
_lib.mq_destroy.argtypes = [ctypes.c_void_p]

_lib.mq_reset.restype  = None
_lib.mq_reset.argtypes = [ctypes.c_void_p]

_lib.mq_apply_gate.restype  = None
_lib.mq_apply_gate.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint32,  # gate_id
    ctypes.c_uint32,  # target
    ctypes.c_int32,   # control (-1 = none)
    ctypes.c_float,   # theta
    ctypes.c_float,   # phi
    ctypes.c_float,   # lam
]

_lib.mq_apply_gates.restype  = None
_lib.mq_apply_gates.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(GateParamStruct),
    ctypes.c_uint32,
]

_lib.mq_statevector.restype  = None
_lib.mq_statevector.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]

_lib.mq_set_state.restype  = None
_lib.mq_set_state.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,  # *const f32 (interleaved re,im)
    ctypes.c_uint32,  # len (number of f32 values = 2 * dim)
]

_lib.mq_sample.restype  = ctypes.c_uint32
_lib.mq_sample.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]

_lib.mq_expect.restype  = ctypes.c_double
_lib.mq_expect.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_uint32, ctypes.c_uint32,
]

_lib.mq_adjoint_gradient.restype  = None
_lib.mq_adjoint_gradient.argtypes = [
    ctypes.c_void_p,                        # simulator
    ctypes.c_void_p,                        # pauli_ids
    ctypes.c_void_p,                        # qubits
    ctypes.c_void_p,                        # coeffs
    ctypes.c_uint32,                        # n_terms
    ctypes.c_uint32,                        # n_qubits_per_term
    ctypes.POINTER(GateParamStruct),        # gate_params
    ctypes.c_uint32,                        # n_gates
    ctypes.c_void_p,                        # param_mask
    ctypes.c_void_p,                        # grad_out
]

# ── Thin wrappers ─────────────────────────────────────────────────────────────

def create(n_qubits: int):
    return _lib.mq_create(n_qubits)

def destroy(handle):
    _lib.mq_destroy(handle)

def reset(handle):
    _lib.mq_reset(handle)

def apply_gate(handle, gate_id, target, control=-1, theta=0.0, phi=0.0, lam=0.0):
    _lib.mq_apply_gate(handle, gate_id, target, control, theta, phi, lam)

def apply_gates(handle, ops):
    """Batch-apply a list of GateOp objects in a single GPU command buffer."""
    n = len(ops)
    if n == 0:
        return
    arr = (GateParamStruct * n)()
    for i, op in enumerate(ops):
        arr[i].gate_id = op.gate_id
        arr[i].target  = op.target
        arr[i].control = op.control
        arr[i].theta   = float(op.resolved_theta())
        arr[i].phi     = float(op.phi)
        arr[i].lam     = float(op.lam)
    _lib.mq_apply_gates(handle, arr, n)

def get_statevector(handle, n_qubits) -> np.ndarray:
    dim = 1 << n_qubits
    buf = np.zeros(dim * 2, dtype=np.float32)
    _lib.mq_statevector(handle, buf.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_uint32(dim * 2))
    return buf[0::2] + 1j * buf[1::2]

def set_state(handle, state: np.ndarray):
    """Write a complex128 state vector into the simulator (debug only)."""
    flat = np.zeros(len(state) * 2, dtype=np.float32)
    flat[0::2] = state.real.astype(np.float32)
    flat[1::2] = state.imag.astype(np.float32)
    _lib.mq_set_state(handle, flat.ctypes.data_as(ctypes.c_void_p),
                      ctypes.c_uint32(len(flat)))

def sample(handle, shots: int) -> np.ndarray:
    buf = np.zeros(shots, dtype=np.uint64)
    n = _lib.mq_sample(handle, shots, buf.ctypes.data_as(ctypes.c_void_p))
    return buf[:n]