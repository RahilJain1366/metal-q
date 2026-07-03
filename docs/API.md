# MetalQ API Reference

Complete reference for all public Python classes, functions, and the Rust/Metal backend internals.

---

## Table of Contents

- [Python Package (`metalq`)](#python-package-metalq)
  - [Circuit](#circuit)
  - [Parameter](#parameter)
  - [GateOp](#gateop)
  - [Hamiltonian](#hamiltonian)
  - [PauliTerm](#pauliterm)
  - [RunResult](#runresult)
  - [StatevectorResult](#statevectorresult)
  - [Runner functions](#runner-functions)
  - [FFI bridge (`_ffi`)](#ffi-bridge-_ffi)
  - [PyTorch integration](#pytorch-integration)
- [Rust Backend (`metal_quantum_native`)](#rust-backend-metal_quantum_native)
  - [C-ABI exports (`lib.rs`)](#c-abi-exports-librs)
  - [Simulator (`simulator.rs`)](#simulator-simulatorrs)
  - [Gates (`gates.rs`)](#gates-gatesrs)
  - [Metal backend (`metal_backend.rs`)](#metal-backend-metal_backendrs)
  - [Gradient (`gradient.rs`)](#gradient-gradientrs)
  - [Sampler (`sampler.rs`)](#sampler-samplerrs)
  - [Hamiltonian (`hamiltonian.rs`)](#hamiltonian-hamiltonianrs)
- [Metal GPU Kernels](#metal-gpu-kernels)
  - [gates.metal](#gatesmetal)
  - [two_qubit.metal](#two_qubitmetal)
  - [gradient.metal](#gradientmetal)
- [Build System (`build.rs`)](#build-system-buildrs)

---

## Python Package (`metalq`)

### `metalq/__init__.py`

Public surface of the package. Re-exports the following names:

```python
from metalq import (
    Circuit,
    Parameter,
    Hamiltonian, X, Y, Z, I,
    RunResult, StatevectorResult,
    run, statevector, expect,
)
```

---

### Circuit

**File:** `metalq/circuit.py`

The central data structure. Holds a list of `GateOp` objects and provides a fluent builder API.

```python
class Circuit:
    def __init__(self, n_qubits: int)
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `n_qubits` | `int` | Number of qubits |
| `ops` | `List[GateOp]` | Ordered list of gate operations |

#### Single-Qubit Gates

All single-qubit gate methods append a `GateOp` to `self.ops` and return `self` for chaining.

| Method | Gate ID | Description |
|--------|---------|-------------|
| `h(qubit)` | 0 | Hadamard — creates superposition |
| `x(qubit)` | 1 | Pauli-X — bit flip |
| `y(qubit)` | 2 | Pauli-Y — bit + phase flip |
| `z(qubit)` | 3 | Pauli-Z — phase flip on `\|1⟩` |
| `t(qubit)` | 11 | T gate — `e^(iπ/4)` phase on `\|1⟩` |
| `s(qubit)` | 12 | S gate — `i` phase on `\|1⟩` |
| `rx(theta, qubit)` | 4 | Rotation around X-axis by angle `theta` |
| `ry(theta, qubit)` | 5 | Rotation around Y-axis by angle `theta` |
| `rz(theta, qubit)` | 6 | Rotation around Z-axis by angle `theta` |
| `u(theta, phi, lam, qubit)` | 7 | General SU(2) gate with 3 Euler angles |

`theta` may be a `float` or a `Parameter` object for parameterized circuits.

#### Two-Qubit Gates

| Method | Gate ID | Description |
|--------|---------|-------------|
| `cx(control, target)` | 8 | CNOT — flips target if control is `\|1⟩` |
| `cz(control, target)` | 9 | CZ — phase-flips `\|11⟩` amplitude |
| `swap(q0, q1)` | 10 | SWAP — exchanges amplitudes of `\|01⟩` and `\|10⟩` |

#### Introspection Methods

**`parameters() → List[Parameter]`**  
Returns all `Parameter` objects found in `self.ops`. Order matches the order gates were added.

**`bind_parameters(values: dict) → Circuit`**  
Returns a new `Circuit` with each `Parameter` whose `.name` is a key in `values` replaced by the corresponding float. Unrecognised parameters are left as-is.

```python
qc = Circuit(1)
theta = Parameter('theta')
qc.rx(theta, 0)
bound = qc.bind_parameters({'theta': 3.14159})
```

**`depth() → int`**  
Computes the circuit depth using a greedy left-to-right layer assignment. Each gate is placed in the earliest layer where all its qubits are free.

**`__repr__`**  
Returns `Circuit(n_qubits=N, depth=D, gates=G)`.

---

### Parameter

**File:** `metalq/gate.py`

A named symbolic trainable angle backed by a `sympy.Symbol`.

```python
class Parameter:
    def __init__(self, name: str)
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Symbolic name |
| `symbol` | `sympy.Symbol` | SymPy symbol for algebraic manipulation |

**`bind(value: float) → Parameter`**  
Sets the numerical value and returns `self`.

**`value → float`** (property)  
Returns the bound float. Raises `ValueError` if unbound.

```python
p = Parameter('alpha')
p.bind(1.57)
print(p.value)   # 1.57
```

---

### GateOp

**File:** `metalq/gate.py`

Immutable dataclass representing a single gate application.

```python
@dataclass
class GateOp:
    name:    str        # e.g. "H", "RX", "CNOT"
    target:  int        # target qubit index
    control: int = -1   # control qubit (-1 = single-qubit gate)
    theta:   object = 0.0   # float or Parameter
    phi:     float = 0.0
    lam:     float = 0.0
```

**`gate_id → int`** (property)  
Looks up the integer gate ID from `GATE_IDS`.

**`resolved_theta() → float`**  
Returns `theta.value` if `theta` is a `Parameter`, otherwise `float(theta)`.

**`is_parameterized() → bool`**  
`True` if `theta` is a `Parameter` instance.

#### Gate ID Map

```python
GATE_IDS = {
    'H': 0, 'X': 1, 'Y': 2, 'Z': 3,
    'RX': 4, 'RY': 5, 'RZ': 6,
    'U': 7, 'CNOT': 8, 'CZ': 9, 'SWAP': 10,
    'T': 11, 'S': 12,
}
```

These IDs must stay in sync with the constants in `gates.rs` and the `switch` statements in the Metal kernels.

---

### Hamiltonian

**File:** `metalq/hamiltonian.py`

Represents a Pauli sum operator `H = Σᵢ cᵢ Pᵢ` where each `Pᵢ` is a tensor product of Pauli operators.

```python
@dataclass
class Hamiltonian:
    terms: List[PauliTerm]
```

**Operators:**

| Method | Returns |
|--------|---------|
| `H1 + H2` | New `Hamiltonian` concatenating both term lists |
| `H * scalar` | New `Hamiltonian` with all coefficients scaled |
| `scalar * H` | Same (via `__rmul__`) |

**Convenience constructors** (module-level functions):

```python
Z(qubit: int) → Hamiltonian   # single-term H = 1.0 * Z_q
X(qubit: int) → Hamiltonian
Y(qubit: int) → Hamiltonian
I(qubit: int) → Hamiltonian
```

**Example:**

```python
from metalq import X, Y, Z
H = -0.5 * Z(0) - 0.5 * Z(1) + 0.25 * X(0) + 0.25 * X(1)
```

**`_pack_hamiltonian(h: Hamiltonian) → (pauli_ids, qubits, coeffs)`** (internal)  
Converts the Hamiltonian into three flat NumPy arrays suitable for passing through FFI:
- `pauli_ids`: `uint8` array, shape `(n_terms, max_ops_per_term)`
- `qubits`: `uint32` array, shape `(n_terms, max_ops_per_term)`
- `coeffs`: `float64` array, shape `(n_terms,)`

Pauli encoding: `I=0, X=1, Y=2, Z=3`.

---

### PauliTerm

**File:** `metalq/hamiltonian.py`

One term in a Pauli sum: `coeff · (P₀ ⊗ P₁ ⊗ ...)`.

```python
@dataclass
class PauliTerm:
    coeff: complex
    ops:   List[Tuple[str, int]]   # e.g. [("Z", 0), ("X", 1)]
```

**`term1 * term2`**  
Returns a new `PauliTerm` with multiplied coefficients and concatenated operator lists.

---

### RunResult

**File:** `metalq/result.py`

Returned by `run()`. Contains shot-based measurement outcomes.

```python
@dataclass
class RunResult:
    counts:   Dict[str, int]   # bitstring → count
    shots:    int
    n_qubits: int
```

| Method | Returns | Description |
|--------|---------|-------------|
| `most_probable()` | `str` | Bitstring with the highest count |
| `probabilities()` | `Dict[str, float]` | Normalised count dict (`count / shots`) |
| `__repr__` | `str` | Top-5 outcomes by count |

Bitstrings are zero-padded to `n_qubits` characters with qubit 0 as the least-significant bit.

---

### StatevectorResult

**File:** `metalq/result.py`

Returned by `statevector()`. Wraps the complex amplitude vector.

```python
@dataclass
class StatevectorResult:
    vector:   np.ndarray   # complex128, shape (2^n_qubits,)
    n_qubits: int
```

| Method | Returns | Description |
|--------|---------|-------------|
| `probabilities()` | `np.ndarray` | `\|amplitude\|²` for each basis state |

Index ordering: `vector[i]` is the amplitude of the basis state whose binary representation is `i`, with qubit 0 as the least-significant bit.

---

### Runner Functions

**File:** `metalq/runner.py`

These are the three top-level entry points into the simulator.

#### `run(circuit, shots=1024, backend="mps") → RunResult`

Executes `circuit` and returns shot-sampled measurement counts.

1. Calls `_execute(circuit)` to allocate a simulator and apply all gates via FFI.
2. Calls `_ffi.sample(h, shots)` to draw bitstring samples from the final statevector distribution.
3. Formats each sample as a zero-padded binary string.
4. Destroys the simulator handle.

#### `statevector(circuit) → StatevectorResult`

Returns the exact complex statevector after applying all gates.

1. Calls `_execute(circuit)`.
2. Calls `_ffi.get_statevector(h, n_qubits)` which copies the GPU-side buffer back as interleaved `float32` pairs, then recombines to `complex128`.
3. Destroys the simulator handle.

#### `expect(circuit, hamiltonian) → float`

Computes `⟨ψ|H|ψ⟩` exactly (no sampling).

1. Calls `_execute(circuit)` to get `|ψ⟩`.
2. Packs the Hamiltonian via `_pack_hamiltonian`.
3. Calls `mq_expect` via FFI which runs the expectation value loop on the CPU in Rust.
4. Destroys the simulator handle.

#### Internal: `_execute(circuit) → int`

Allocates a simulator (`mq_create`), iterates over `circuit.ops`, and calls `mq_apply_gate` for each. Returns the opaque handle.

---

### FFI bridge (`_ffi`)

**File:** `metalq/_ffi.py`

Low-level ctypes bridge. Not part of the public API — use `runner.py` functions instead.

**Library loading:**  
Searches for `libmetalq_native.dylib` in:
1. `<package_root>/../target/release/libmetalq_native.dylib` (in-tree build)
2. `<package_root>/libmetalq_native.dylib` (bundled install)

Raises `FileNotFoundError` if neither is found.

**Thin wrapper functions:**

| Function | Wraps | Description |
|----------|-------|-------------|
| `create(n_qubits)` | `mq_create` | Returns opaque handle |
| `destroy(handle)` | `mq_destroy` | Frees simulator memory |
| `reset(handle)` | `mq_reset` | Resets to `\|0...0⟩` |
| `apply_gate(handle, gate_id, target, control, theta, phi, lam)` | `mq_apply_gate` | Dispatches one gate |
| `get_statevector(handle, n_qubits)` | `mq_statevector` | Returns `complex128` ndarray |
| `sample(handle, shots)` | `mq_sample` | Returns `uint64` ndarray of outcomes |

`get_statevector` allocates a `float32` buffer of length `2 * 2^n_qubits`, reads interleaved (real, imag) pairs, and recombines: `buf[0::2] + 1j * buf[1::2]`.

---

### PyTorch Integration

**Files:** `metalq/torch/layer.py`, `metalq/torch/function.py`

#### `QuantumLayer`

```python
class QuantumLayer(nn.Module):
    def __init__(self, circuit: Circuit, hamiltonian: Hamiltonian,
                 backend_name: str = "mps")
```

A `torch.nn.Module` that wraps a parameterized circuit and a Hamiltonian. The module's trainable parameter tensor `self.thetas` has one entry per `Parameter` in `circuit.parameters()`, in the same order.

**`forward(x=None) → torch.Tensor`**  
Returns `⟨H⟩` as a scalar tensor. Delegates to `QuantumFunction.apply(...)`.

```python
layer = QuantumLayer(circuit, hamiltonian)
loss = layer()          # computes ⟨H⟩
loss.backward()         # computes ∂⟨H⟩/∂θᵢ via adjoint method
```

#### `QuantumFunction`

```python
class QuantumFunction(torch.autograd.Function)
```

Custom autograd Function. Not normally instantiated directly.

**`forward(ctx, thetas, circuit, hamiltonian, param_names)`**  
Binds `thetas` into the circuit and calls `expect(bound_circuit, hamiltonian)`. Saves tensors and context for backward.

**`backward(ctx, grad_output)`**  
Calls `_adjoint_grad` to compute `∂⟨H⟩/∂θᵢ` for each parameterized gate, then returns `grad_output * grads`.

#### `_adjoint_grad(circuit, hamiltonian, param_names, theta_values)`

Internal. Builds a `GateParamStruct` array and `param_mask`, calls `mq_adjoint_gradient` via FFI, and returns only the gradient values for parameterized gates (those where `param_mask[i] == 1`).

---

## Rust Backend (`metal_quantum_native`)

**Crate type:** `cdylib` — compiled to `libmetalq_native.dylib`

**Dependencies:**
- `metal = "0.29"` — Apple Metal bindings
- `num-complex = "0.4"` — `Complex<f32>` statevector elements
- `bytemuck = "1.14"` — safe casting for buffer data
- `thiserror = "1"` — error types
- `libc = "0.2"`, `objc = "0.2"` — low-level system interop

---

### C-ABI Exports (`lib.rs`)

All exported symbols use `#[no_mangle]` and `extern "C"` so they are callable from Python ctypes.

```rust
pub extern "C" fn mq_create(n_qubits: u32) -> *mut c_void
```
Allocates a `Simulator` on the heap, returns an opaque pointer.

```rust
pub unsafe extern "C" fn mq_destroy(ptr: *mut c_void)
```
Drops the `Box<Simulator>`, freeing all memory.

```rust
pub unsafe extern "C" fn mq_reset(ptr: *mut c_void)
```
Calls `Simulator::reset()` to reinitialise the statevector to `|0...0⟩`.

```rust
pub unsafe extern "C" fn mq_apply_gate(
    ptr: *mut c_void,
    gate_id: u32, target: u32, control: i32,
    theta: f32, phi: f32, lam: f32,
)
```
Delegates to `Simulator::apply_gate`.

```rust
pub unsafe extern "C" fn mq_statevector(
    ptr: *mut c_void, out: *mut f32, len: u32,
)
```
Writes `min(len/2, 2^n)` complex amplitudes as interleaved `(re, im)` `f32` pairs into `out`.

```rust
pub unsafe extern "C" fn mq_expect(
    ptr: *mut c_void,
    pauli_ids: *const u8, qubits: *const u32, coeffs: *const f64,
    n_terms: u32, n_qubits_per_term: u32,
) -> f64
```
Calls `hamiltonian::expect` and returns the real part of `⟨ψ|H|ψ⟩`.

```rust
pub unsafe extern "C" fn mq_sample(
    ptr: *mut c_void, shots: u32, out: *mut u64,
) -> u32
```
Calls `sampler::sample`, writes outcomes into `out`, returns actual count.

```rust
pub unsafe extern "C" fn mq_adjoin_gradient(...)
```
Calls `gradient::adjoint` for the adjoint differentiation backward pass.

#### `GateParam` struct

```rust
#[repr(C)]
pub struct GateParam {
    pub gate_id: u32,
    pub target:  u32,
    pub control: i32,
    pub theta:   f32,
    pub phi:     f32,
    pub lam:     f32,
}
```

Passed as an array in the adjoint gradient call. Layout must match the Python-side `ctypes` structure.

---

### Simulator (`simulator.rs`)

```rust
pub struct Simulator {
    pub n_qubits: usize,
    pub state:    Vec<Complex<f32>>,
    backend:      MetalBackend,
}
```

**`new(n_qubits) → Simulator`**  
Allocates `state` of length `2^n_qubits`, sets `state[0] = 1.0 + 0i` (`|0...0⟩`), constructs `MetalBackend`.

**`reset(&mut self)`**  
Fills `state` with zeros, then sets `state[0] = 1.0 + 0i`.

**`apply_gate(&mut self, gate_id, target, control, theta, phi, lam)`**  
Delegates to `MetalBackend::apply_gate`, passing `&mut self.state` directly.

**`statevector(&self) → &[Complex<f32>]`**  
Returns a slice reference to `self.state`.

---

### Gates (`gates.rs`)

Gate ID constants used across Rust and Metal:

```rust
pub const H:    u32 = 0;
pub const X:    u32 = 1;
pub const Y:    u32 = 2;
pub const Z:    u32 = 3;
pub const RX:   u32 = 4;
pub const RY:   u32 = 5;
pub const RZ:   u32 = 6;
pub const U:    u32 = 7;
pub const CNOT: u32 = 8;
pub const CZ:   u32 = 9;
pub const SWAP: u32 = 10;
pub const T:    u32 = 11;
pub const S:    u32 = 12;
```

**`is_two_qubit(gate_id: u32) → bool`**  
Returns `true` for `CNOT`, `CZ`, `SWAP`. Used to halve the thread count dispatch.

**`kernel_name(gate_id: u32) → &'static str`**  
Maps gate ID to the Metal kernel function name:
- All single-qubit gates → `"apply_gate"`
- `CNOT` → `"apply_cnot"`
- `CZ` → `"apply_cz"`
- `SWAP` → `"apply_swap"`

---

### Metal Backend (`metal_backend.rs`)

```rust
pub struct MetalBackend {
    device: Device,
}
```

Holds a reference to the system default Metal device. Printed on construction: `[MetalQ] GPU backend: <device name>`.

**`new() → MetalBackend`**  
Calls `Device::system_default()`. Panics if no Metal device is found.

**`apply_gate(&mut self, state, n_qubits, gate_id, target, control, theta, phi, lam)`**

Full GPU dispatch sequence for one gate:

1. Load `MetalKernels.metallib` from the filesystem.
2. Look up the kernel function by name (`gates::kernel_name(gate_id)`).
3. Create a compute pipeline state.
4. Create a command queue and command buffer.
5. Upload `state` to a shared `MTLBuffer` (`StorageModeShared` — no explicit copy needed on Apple Silicon).
6. Upload a `GateParams` struct to a second buffer.
7. Set thread count: `N/2` for single-qubit, `N/4` for two-qubit gates.
8. Dispatch with `MTLSize(n_threads, 1, 1)`.
9. Commit and `wait_until_completed()`.
10. Read the result buffer back into `state` via raw pointer dereference.

**Note:** A new library, pipeline, and command queue are created for every gate call. This is a correctness-first implementation; batching multiple gates into one dispatch would be a significant performance optimisation.

---

### Gradient (`gradient.rs`)

Implements the adjoint differentiation algorithm entirely on the CPU.

**`unsafe fn adjoint(sim, pauli_ids, qubits, coeffs, n_terms, n_qpt, gate_params, param_mask, grad_out)`**

Steps:
1. **Forward pass** — copy `sim.statevector()` and apply each gate in `gate_params` using `apply_gate_cpu`.
2. **Initialise `|λ⟩`** — call `apply_hamiltonian_cpu(&psi, ...)` to get `H|ψ⟩`.
3. **Backward pass** — iterate gates in reverse:
   - If `param_mask[i] == 1`: compute `dG/dθ |ψ⟩` via `apply_gate_derivative_cpu`, then `grad[i] = 2 · Re⟨λ|dG/dθ|ψ⟩`.
   - Apply `G†` to both `|ψ⟩` and `|λ⟩` via `apply_gate_dagger_cpu`.

#### Helper Functions

**`apply_gate_cpu(state, n_qubits, gp)`**  
CPU implementation of gate application. Iterates over `dim/2` pairs `(lo, hi)`. Implements H, X, Y, Z, RX, RY, RZ, plus the control qubit guard. Used in the adjoint forward and backward passes.

**`apply_gate_dagger_cpu(state, n_qubits, gp)`**  
Applies `G†` (conjugate transpose). For self-adjoint gates (H, X, Y, Z): same as `G`. For rotation gates: negate `theta`, `phi`, `lam`.

**`apply_gate_derivative_cpu(state, n_qubits, gp)`**  
Applies `dG/dθ` to `state`. For rotation gates:
- `dRX/dθ = -i/2 · X · RX(θ)` → apply RX, then X, then multiply by `-i/2`.
- `dRY/dθ = -i/2 · Y · RY(θ)` → apply RY, then Y, then multiply by `-i/2`.
- `dRZ/dθ = -i/2 · Z · RZ(θ)` → apply RZ, then Z, then multiply by `-i/2`.
- Non-rotation gates: no-op.

**`unsafe fn apply_hamiltonian_cpu(psi, pauli_ids, qubits, coeffs, n_terms, n_qpt, dim) → Vec<Complex<f32>>`**  
Computes `H|ψ⟩` by iterating over each Pauli term, each basis state, and each Pauli operator in the term. Applies the phase and bit-flip rules for X, Y, Z, accumulates into result.

---

### Sampler (`sampler.rs`)

**`pub fn sample(sim: &Simulator, shots: usize) → Vec<u64>`**

1. Computes probability `|amplitude|²` for each basis state.
2. Builds a cumulative probability distribution.
3. Draws `shots` uniform random numbers in `[0, 1)` via `random_f64()`.
4. For each draw, binary-searches the CDF (`partition_point`) to find the outcome index.
5. Returns `Vec<u64>` of length `shots`.

**`fn random_f64() → f64`**  
Thread-local xorshift64 PRNG. Seeded from `SystemTime::now().subsec_nanos()`. Uses the standard xorshift sequence `x ^= x << 13; x ^= x >> 7; x ^= x << 17`. Normalises to `[0.0, 1.0)` via `(x >> 11) as f64 / (1u64 << 53) as f64`.

---

### Hamiltonian (`hamiltonian.rs`)

**`pub unsafe fn expect(sim, pauli_ids, qubits, coeffs, n_terms, n_qpt) → f64`**

Computes `⟨ψ|H|ψ⟩` exactly:

```
total = Σ_t  coeff[t] · Re(⟨ψ|P_t|ψ⟩)
```

For each Pauli term `t` and each basis state `i`:
- Determine the output index `j` and phase after applying each Pauli operator in the term.
- Accumulate `conj(ψ[j]) · phase · ψ[i]` into `term_val`.
- Add `coeff · term_val.re` to `total`.

Pauli action rules:
- **I**: `j = i`, `phase = 1`
- **X**: `j = i XOR (1 << qubit)`, `phase = 1`
- **Y**: `j = i XOR (1 << qubit)`, `phase = i if qubit bit is 0, else -i`
- **Z**: `j = i`, `phase = -1 if qubit bit is 1, else 1`

---

## Metal GPU Kernels

Compiled from `.metal` source → `.air` intermediate → `MetalKernels.metallib` at build time.

### `shaders/gates.metal`

Implements `kernel void apply_gate(...)` for all single-qubit gates (gate IDs 0–7, 11, 12).

**Shared struct:**
```metal
struct GateParams {
    uint gate_id; uint target; int control;
    float theta; float phi; float lam;
};
```

**Thread mapping** — each thread `gid ∈ [0, N/2)` computes a unique `(lo, hi)` pair:
```metal
uint lo = ((gid >> n) << (n + 1u)) | (gid & ((1u << n) - 1u));
uint hi = lo | (1u << n);
```

This inserts a zero at bit position `n` of `gid`, so threads cover all amplitude pairs without aliasing.

**Control guard:**
```metal
if (params.control >= 0) {
    uint ctrl = (uint)params.control;
    if (!((lo >> ctrl) & 1u)) return;
}
```

**Gate implementations** (switch on `gate_id`):

| ID | Gate | Key formula |
|----|------|------------|
| 0 | H | `state[lo] = (a+b)/√2; state[hi] = (a-b)/√2` |
| 1 | X | `state[lo] = b; state[hi] = a` |
| 2 | Y | `state[lo] = (-i)b; state[hi] = (i)a` |
| 3 | Z | `state[hi] = -b` |
| 4 | RX(θ) | `[[c, -is], [-is, c]]` where `c=cos(θ/2), s=sin(θ/2)` |
| 5 | RY(θ) | `[[c, -s], [s, c]]` (real-valued) |
| 6 | RZ(θ) | `[[e^(-iθ/2), 0], [0, e^(iθ/2)]]` |
| 7 | U(θ,φ,λ) | Full 3-parameter SU(2) |
| 11 | T | `state[hi] = e^(iπ/4) · b` (cos=sin=1/√2) |
| 12 | S | `state[hi] = i · b` (rotate 90°) |

The statevector is stored as `device float2*` where `.x` = real part, `.y` = imaginary part.

---

### `shaders/two_qubit.metal`

Implements `apply_cnot`, `apply_cz`, `apply_swap` for two-qubit operations.

**Thread mapping:**  
Each thread `gid ∈ [0, N/4)` owns a block of 4 amplitudes. The `base_idx` helper inserts zeros at both qubit bit positions:

```metal
static inline uint insert_zero(uint v, uint bit) {
    uint lo_mask = (1u << bit) - 1u;
    return ((v & ~lo_mask) << 1u) | (v & lo_mask);
}

static inline uint base_idx(uint gid, uint ctrl, uint tgt) {
    uint q0 = min(ctrl, tgt);
    uint q1 = max(ctrl, tgt);
    return insert_zero(insert_zero(gid, q0), q1);
}
```

**`apply_cnot`:** swaps `state[i10]` and `state[i11]` (the two amplitudes where control qubit is 1).

**`apply_cz`:** negates `state[i11]` (phase-flip when both qubits are 1).

**`apply_swap`:** swaps `state[i01]` and `state[i10]` (the two amplitudes where qubit states differ).

---

### `shaders/gradient.metal`

Implements `kernel void adjoint_gradient_step(...)` for GPU-accelerated gradient accumulation.

**Purpose:** Computes `2·Re⟨λ|dG/dθ|ψ⟩` for one gate, with a tree-reduction within each threadgroup to avoid atomic writes.

**Buffers:**
- `buffer(0)`: `psi` — current forward state `float2*`
- `buffer(1)`: `lam` — backward lambda state `float2*`
- `buffer(2)`: `GateParams`
- `buffer(3)`: `grad_buf` — output partial sums, one `float` per threadgroup

**Algorithm per thread:**
1. Compute `(lo, hi)` pair.
2. Load `psi[lo], psi[hi], lam[lo], lam[hi]`.
3. Switch on `gate_id` (4=RX, 5=RY, 6=RZ):
   - Compute `dlo, dhi = (dG/dθ)|ψ⟩` at this pair.
   - `partial = Re(lam[lo]·dlo) + Re(lam[hi]·dhi)`.
4. Tree-reduce `partial` within threadgroup → write `2.0 * tg_partial[0]` to `grad_buf[tg_id]`.

The Rust caller sums `grad_buf` to get the final scalar gradient for this gate.

**Note:** The current `gradient.rs` CPU implementation does not yet call this kernel — it performs adjoint differentiation entirely on the CPU. This shader is the intended GPU path for a future optimisation.

---

## Build System (`build.rs`)

**File:** `metal_quantum_native/build.rs`

Runs at Cargo build time via the standard `build.rs` convention.

**Steps:**

1. Resolve the workspace root (one directory above the crate manifest).
2. For each shader (`gates`, `two_qubit`, `gradient`):
   - Skip with a warning if the `.metal` file is missing.
   - Compile `shaders/<name>.metal` → `<OUT_DIR>/<name>.air`:
     ```bash
     xcrun -sdk macosx metal -c shaders/<name>.metal -o <name>.air
     ```
3. Link all `.air` files into `MetalKernels.metallib`:
   ```bash
   xcrun -sdk macosx metallib gates.air two_qubit.air gradient.air -o MetalKernels.metallib
   ```
4. Copy the `.metallib` to the workspace root so `device.new_library_with_file("MetalKernels.metallib")` resolves when the process runs from the workspace root.
5. Emit `cargo:rerun-if-changed=` directives for each shader and for `build.rs` itself.

Requires Xcode (or Xcode Command Line Tools) to be installed for `xcrun`.