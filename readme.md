# MetalQ

A quantum circuit simulator that runs entirely on Apple GPUs via **Apple Metal**, with a **Rust** native backend and a clean **Python** API.

```
Python API  ──►  ctypes FFI  ──►  Rust dylib  ──►  Metal GPU kernels
                              (libmetalq_native.dylib)   (MetalKernels.metallib)
```

---

## Performance

Benchmarked on Apple M1 Pro. Each number is the median of 5 runs.
Circuit: 4 alternating layers of random single-qubit rotations followed by a CNOT ladder.

**Random rotation + CNOT circuit (statevector simulation)**

| Qubits | MetalQ   | Qiskit Aer | PennyLane | Aer / MetalQ |
|-------:|:--------:|:----------:|:---------:|:------------:|
| 4      | 1.3 ms   | 30.8 ms    | 4.1 ms    | 24×          |
| 8      | 2.5 ms   | 31.5 ms    | 8.5 ms    | 13×          |
| 12     | 3.3 ms   | 34.3 ms    | 17.4 ms   | 10×          |
| 16     | 12.8 ms  | 57.3 ms    | 51.2 ms   | 4.5×         |
| 20     | 41.3 ms  | 311.7 ms   | 1.22 s    | 7.5×         |

**QFT circuit (statevector simulation)**

| Qubits | MetalQ   | Qiskit Aer | PennyLane | Aer / MetalQ |
|-------:|:--------:|:----------:|:---------:|:------------:|
| 4      | 4.5 ms   | 28.5 ms    | 0.3 ms    | 6×           |
| 8      | 1.6 ms   | 30.3 ms    | 3.4 ms    | 19×          |
| 12     | 3.0 ms   | 33.7 ms    | 12.3 ms   | 11×          |
| 16     | 12.5 ms  | 47.3 ms    | 50.0 ms   | 3.8×         |
| 20     | 38.5 ms  | 167.6 ms   | 846 ms    | 4.3×         |

**Note on small circuits:** at 4 qubits the Metal kernel dispatch adds ~1–4 ms of fixed overhead, which dominates when the state vector is only 16 amplitudes. MetalQ's advantage grows with qubit count as the O(2^n) work amortises that overhead. Run `python benchmarks/bench.py` to reproduce on your machine.

---

## Overview

MetalQ dispatches every gate operation as a GPU compute kernel on the Metal device, keeping the statevector in GPU-shared memory throughout a circuit execution. Gradient computation uses the adjoint differentiation method, and PyTorch integration exposes the simulator as a standard `nn.Module` for hybrid quantum-classical learning.

See [`docs/adjoint_method.md`](docs/adjoint_method.md) for a technical explanation of why adjoint differentiation was chosen over parameter-shift and how it is implemented here.

---

## Requirements

| Dependency | Version | Notes |
|-----------|---------|-------|
| macOS | 13+ | Metal 3 GPU required |
| Xcode / Command Line Tools | 15+ | Provides `xcrun metal` and `xcrun metallib` |
| Rust + Cargo | 1.75+ | `rustup` recommended |
| Python | 3.10+ | |
| NumPy | 1.26+ | |
| SymPy | 1.12+ | For symbolic `Parameter` |
| PyTorch | 2.2+ | Optional, for `QuantumLayer` |

---

## Installation

### 1. Build the Rust / Metal backend

```bash
cd metal_quantum_native
cargo build --release
```

The build script (`build.rs`) automatically:
- Compiles `shaders/gates.metal`, `shaders/two_qubit.metal`, and `shaders/gradient.metal` to `.air` (Metal IR)
- Links them into `MetalKernels.metallib` at the workspace root
- Copies the `.metallib` to the workspace root so the runtime can locate it

### 2. Install the Python package

```bash
pip install -e ".[dev]"          # editable install with dev tools
pip install -e ".[torch]"        # add PyTorch integration
pip install -e ".[torch,qiskit]" # add Qiskit adapter
```

### 3. Verify the installation

```bash
python verify.py
```

Expected output:
```
==================================================
MetalQ Verification Suite
==================================================

Test 1: Bell State
  Counts: {'00': 512, '11': 488}
  PASSED

Test 2: Statevector of H|0⟩
  |ψ⟩ = [0.70710678+0.j 0.70710678+0.j]
  PASSED
...
All tests passed ✓
```

---

## Quick Start

### Bell State

```python
from metalq import Circuit, run

qc = Circuit(2)
qc.h(0).cx(0, 1)

result = run(qc, shots=1024)
print(result)
# RunResult(shots=1024, top=[('00', 513), ('11', 511)])
```

### Statevector Simulation

```python
from metalq import Circuit, statevector
import numpy as np

qc = Circuit(1)
qc.h(0)

sv = statevector(qc)
print(sv.vector)            # [0.707+0j, 0.707+0j]
print(sv.probabilities())   # [0.5, 0.5]
```

### Expectation Values

```python
from metalq import Circuit, expect, Z

qc = Circuit(1)
qc.x(0)              # prepare |1⟩

val = expect(qc, Z(0))
print(val)           # -1.0  (|1⟩ is the -1 eigenstate of Z)
```

### Parameterized Circuits

```python
from metalq import Circuit, Parameter, statevector
import numpy as np

theta = Parameter('theta')
qc = Circuit(1)
qc.rx(theta, 0)

bound = qc.bind_parameters({'theta': np.pi})
sv = statevector(bound)
print(sv.probabilities())  # [~0.0, ~1.0]  (RX(π)|0⟩ = |1⟩)
```

### PyTorch Hybrid Model

```python
import torch
from metalq import Circuit, Parameter
from metalq.torch import QuantumLayer
from metalq import Z

theta = Parameter('theta')
qc = Circuit(1)
qc.ry(theta, 0)

layer = QuantumLayer(qc, Z(0))   # wraps circuit + Hamiltonian as nn.Module
optimizer = torch.optim.Adam(layer.parameters(), lr=0.1)

for step in range(50):
    optimizer.zero_grad()
    loss = layer()           # forward: computes ⟨Z⟩
    loss.backward()          # backward: adjoint gradient through Metal
    optimizer.step()

print(f"Final ⟨Z⟩: {layer().item():.4f}")   # should converge to -1.0
```

---

## Gate Reference

### Single-Qubit Gates

| Method | Gate | Matrix |
|--------|------|--------|
| `qc.h(q)` | Hadamard | `(1/√2) [[1,1],[1,-1]]` |
| `qc.x(q)` | Pauli-X | `[[0,1],[1,0]]` |
| `qc.y(q)` | Pauli-Y | `[[0,-i],[i,0]]` |
| `qc.z(q)` | Pauli-Z | `[[1,0],[0,-1]]` |
| `qc.t(q)` | T gate | `[[1,0],[0,e^(iπ/4)]]` |
| `qc.s(q)` | S gate | `[[1,0],[0,i]]` |
| `qc.rx(θ, q)` | RX(θ) | `[[cos θ/2, -i sin θ/2], [-i sin θ/2, cos θ/2]]` |
| `qc.ry(θ, q)` | RY(θ) | `[[cos θ/2, -sin θ/2], [sin θ/2, cos θ/2]]` |
| `qc.rz(θ, q)` | RZ(θ) | `[[e^(-iθ/2), 0], [0, e^(iθ/2)]]` |
| `qc.u(θ,φ,λ, q)` | U gate | General SU(2) |

### Two-Qubit Gates

| Method | Gate | Action |
|--------|------|--------|
| `qc.cx(ctrl, tgt)` | CNOT | `\|c,t⟩ → \|c, t⊕c⟩` |
| `qc.cz(ctrl, tgt)` | CZ | Phase flip `\|11⟩` |
| `qc.swap(q0, q1)` | SWAP | Exchange amplitudes `\|01⟩ ↔ \|10⟩` |

All gate methods return `self` for chaining: `qc.h(0).cx(0, 1).rz(np.pi/4, 1)`.

---

## Project Structure

```
metal-q/
├── Cargo.toml                    # Rust workspace root
├── pyproject.toml                # Python package config
├── LICENSE                       # MIT
├── verify.py                     # End-to-end smoke test suite
│
├── benchmarks/
│   └── bench.py                  # MetalQ vs Qiskit Aer vs PennyLane
│
├── metal_quantum_native/         # Rust crate → libmetalq_native.dylib
│   ├── Cargo.toml
│   ├── build.rs                  # Compiles .metal → .air → .metallib
│   └── src/
│       ├── lib.rs                # C-ABI FFI surface (mq_* exports)
│       ├── simulator.rs          # Statevector allocator & gate dispatcher
│       ├── gates.rs              # Gate ID constants & kernel name mapping
│       ├── metal_backend.rs      # Apple Metal GPU dispatch (metal-rs)
│       ├── gradient.rs           # Adjoint differentiation
│       ├── sampler.rs            # Shot-based measurement (xorshift64 RNG)
│       └── hamiltonian.rs        # PauliSum expectation value ⟨ψ|H|ψ⟩
│
├── shaders/
│   ├── gates.metal               # Single-qubit gate kernels (H,X,Y,Z,RX,RY,RZ,U,T,S)
│   ├── two_qubit.metal           # Two-qubit kernels (CNOT, CZ, SWAP)
│   └── gradient.metal            # Adjoint gradient kernels (RX, RY, RZ)
│
├── metalq/                       # Python package
│   ├── __init__.py               # Public API surface
│   ├── circuit.py                # Circuit class
│   ├── gate.py                   # GateOp, Parameter
│   ├── hamiltonian.py            # Hamiltonian, X(), Y(), Z(), I()
│   ├── result.py                 # RunResult, StatevectorResult
│   ├── runner.py                 # run(), statevector(), expect()
│   ├── _ffi.py                   # ctypes bridge to Rust dylib
│   └── torch/
│       ├── layer.py              # QuantumLayer (nn.Module)
│       └── function.py           # QuantumFunction (torch.autograd.Function)
│
├── docs/
│   ├── API.md                    # Full API reference
│   └── adjoint_method.md         # Technical note: adjoint vs parameter-shift
│
└── tests/
    ├── smoke_test.py
    ├── test_classical.py
    └── test_gates.py
```

---

## Architecture Deep Dive

### GPU Execution Model

Each gate application creates a Metal compute pass:

1. The statevector (length `2^n`, stored as `float2` complex pairs) is uploaded to a `MTLBuffer` in shared memory.
2. A `GateParams` struct (`gate_id`, `target`, `control`, `theta`, `phi`, `lam`) is passed as a second buffer.
3. The appropriate kernel (`apply_gate`, `apply_cnot`, `apply_cz`, `apply_swap`) is dispatched with:
   - **Single-qubit gates**: `N/2` threads — each thread owns one (lo, hi) amplitude pair.
   - **Two-qubit gates**: `N/4` threads — each thread owns one 4-amplitude block.
4. The result is read back from shared GPU memory into the CPU-side Rust `Vec`.

The bit-insertion trick used in the kernels maps a contiguous thread index `gid ∈ [0, N/2)` to the correct (lo, hi) pair without branching:
```metal
uint lo = ((gid >> n) << (n + 1u)) | (gid & ((1u << n) - 1u));
uint hi = lo | (1u << n);
```

### Adjoint Differentiation

Gradients are computed using the adjoint method (O(1) memory overhead vs. O(n) for parameter shift):

1. **Forward pass**: run the full circuit to get `|ψ⟩`.
2. **Initialize**: `|λ⟩ = H|ψ⟩` (apply Hamiltonian on CPU).
3. **Backward pass** (reverse through gates):
   - At each parameterized gate: `grad[i] = 2 · Re⟨λ | dG/dθ | ψ⟩`
   - Unapply `G†` from both `|ψ⟩` and `|λ⟩`.

For rotation gates: `dRn(θ)/dθ = -i/2 · Pn · Rn(θ)` where `Pn` is the corresponding Pauli.

See [`docs/adjoint_method.md`](docs/adjoint_method.md) for the full derivation and a comparison against parameter-shift.

### FFI Bridge

Python communicates with Rust through `ctypes` (`metalq/_ffi.py`). The C-ABI exported functions are:

| Function | Signature | Description |
|----------|-----------|-------------|
| `mq_create` | `(n_qubits: u32) → *void` | Allocate simulator |
| `mq_destroy` | `(*void) → void` | Free simulator |
| `mq_reset` | `(*void) → void` | Reset to `\|0...0⟩` |
| `mq_apply_gate` | `(*void, gate_id, target, control, θ, φ, λ)` | Apply one gate |
| `mq_statevector` | `(*void, out_f32*, len)` | Copy statevector |
| `mq_sample` | `(*void, shots, out_u64*)` | Sample bit-strings |
| `mq_expect` | `(*void, pauli_ids*, qubits*, coeffs*, n_terms, n_qpt) → f64` | Compute ⟨H⟩ |
| `mq_adjoin_gradient` | `(*void, ..., gate_params*, n_gates, mask*, grad_out*)` | Adjoint gradients |

---

## API Summary

### `Circuit`

```python
Circuit(n_qubits: int)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `.h(q)` | `self` | Hadamard |
| `.x(q)`, `.y(q)`, `.z(q)` | `self` | Pauli gates |
| `.t(q)`, `.s(q)` | `self` | T and S gates |
| `.rx(θ, q)`, `.ry(θ, q)`, `.rz(θ, q)` | `self` | Rotation gates |
| `.u(θ, φ, λ, q)` | `self` | Universal gate |
| `.cx(ctrl, tgt)`, `.cz(ctrl, tgt)` | `self` | Controlled gates |
| `.swap(q0, q1)` | `self` | SWAP gate |
| `.parameters()` | `List[Parameter]` | List of unbound parameters |
| `.bind_parameters(dict)` | `Circuit` | New circuit with floats substituted |
| `.depth()` | `int` | Circuit depth (greedy layer count) |

### Top-Level Functions

```python
run(circuit, shots=1024, backend="mps") → RunResult
statevector(circuit) → StatevectorResult
expect(circuit, hamiltonian) → float
```

### `Parameter`

```python
p = Parameter('theta')
p.bind(3.14)     # returns self
p.value          # raises ValueError if unbound
```

### `Hamiltonian`

```python
H = 0.5 * Z(0) + 0.5 * Z(1) - X(0) * X(1)
```

Supports `+`, `*` (scalar), and `__rmul__`. Build terms with `X(q)`, `Y(q)`, `Z(q)`, `I(q)`.

### `RunResult`

```python
result.counts            # Dict[str, int]  e.g. {'00': 512, '11': 512}
result.shots             # int
result.most_probable()   # str  e.g. '00'
result.probabilities()   # Dict[str, float]
```

### `StatevectorResult`

```python
sv.vector          # np.ndarray complex128, shape (2^n,)
sv.n_qubits        # int
sv.probabilities() # np.ndarray float64, shape (2^n,)
```

---

## Testing

```bash
# Quick smoke test
python verify.py

# Full test suite
pytest tests/

# Benchmark against Qiskit Aer and PennyLane
python benchmarks/bench.py
```

---

## Troubleshooting

**`FileNotFoundError: libmetalq_native.dylib not found`**
Run `cargo build --release` inside `metal_quantum_native/` first.

**`Failed to load Metal library`**
The `MetalKernels.metallib` must be in the working directory when Python is invoked. It is placed at the workspace root by `build.rs`. Run Python from the `metal-q/` directory, or set the working directory accordingly.

**`No Metal device found`**
This simulator requires a Mac with a Metal-capable GPU (all Apple Silicon and most Intel Macs with macOS 13+). It will not run on Linux or Windows.

**Segfault or `EXC_BAD_ACCESS`**
Usually a mismatch between the `ctypes` function signatures in `_ffi.py` and the actual `extern "C"` signatures in `lib.rs`. Check that argument types and order match exactly.

---

## Acknowledgment

I built the Metal compute kernels (`shaders/`), the Rust backend including the simulator, adjoint differentiation, shot sampler, and Hamiltonian expectation engine (`metal_quantum_native/src/`), the Python FFI bridge, circuit abstraction, and PyTorch integration layer from scratch. The project uses `metal-rs` (Rust bindings for Apple Metal), `num-complex` for complex arithmetic, and standard Python packaging tooling. Gate conventions and the adjoint algorithm follow Nielsen & Chuang (2000) and Jones & Gacon (2020).

---

## License

MIT — see [LICENSE](LICENSE).