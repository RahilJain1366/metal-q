# Why I Used the Adjoint Method for Quantum Gradients

When I built the gradient engine for MetalQ I had two realistic options: **parameter-shift** and **adjoint differentiation**. I chose adjoint. This note explains what both methods do, why adjoint is strictly better for a statevector simulator, and what the derivation actually looks like so the choice is not just a citation.

---

## The problem

We want ∂⟨H⟩/∂θ, where ⟨H⟩ = ⟨ψ(θ)|H|ψ(θ)⟩ and |ψ(θ)⟩ = Uₙ(θₙ)…U₁(θ₁)|0⟩.

Both methods produce exact gradients (no finite-difference noise). They differ in how many circuit evaluations they require and how much memory they consume.

---

## Parameter-shift

The parameter-shift rule exploits the fact that rotation gates have a spectrum of ±1/2:

```
∂⟨H⟩/∂θᵢ = [ ⟨H⟩(θᵢ + π/2) − ⟨H⟩(θᵢ − π/2) ] / 2
```

This is exact (not an approximation) and follows from the generator algebra: for Rₙ(θ) = exp(−iθPₙ/2), differentiating gives −iPₙ/2 · Rₙ(θ), and the shift ±π/2 evaluates the two terms of that derivative as separate expectation values.

**Cost:** 2 full circuit executions per parameter. A circuit with *p* parameters requires **2p forward passes** to compute the full gradient.

In a variational quantum eigensolver (VQE) with, say, 50 parameters, that is 100 circuit evaluations per gradient step. For a hardware quantum computer that is unavoidable — you cannot inspect the internal state, so you must re-run the circuit. But for a simulator, it is purely wasteful.

---

## Adjoint differentiation

The adjoint method (Liao, Jones, Gacon et al., 2020; originally described for classical ML as "reverse-mode AD through unitary circuits") needs **one forward pass and one backward pass**, regardless of the number of parameters.

### Derivation

Write the circuit as a product of gates:

```
|ψ⟩ = Uₙ … Uₖ₊₁ Uₖ … U₁ |0⟩
```

The expectation value is ⟨ψ|H|ψ⟩. Its gradient with respect to the angle inside the k-th gate is:

```
∂⟨H⟩/∂θₖ = ⟨ψ| [∂Uₖ/∂θₖ]† (Uₙ…Uₖ₊₁)† H (Uₙ…Uₖ₊₁) ∂Uₖ/∂θₖ |ψₖ₋₁⟩ 
           + h.c.
```

where |ψₖ₋₁⟩ = Uₖ₋₁ … U₁|0⟩ is the state *before* the k-th gate.

Define the **bra-state** (the "lambda vector") after applying all gates up to and including k:

```
⟨λₖ| = ⟨ψ| (Uₙ … Uₖ₊₁)† H (Uₙ … Uₖ₊₁)
```

Then:

```
∂⟨H⟩/∂θₖ = 2 Re⟨λₖ | (∂Uₖ/∂θₖ) | ψₖ₋₁⟩
```

For a rotation gate Rₙ(θ) = exp(−iθPₙ/2):

```
∂Rₙ/∂θ = −i/2 · Pₙ · Rₙ(θ)
```

So the derivative state is (−i/2 · Pₙ)|ψₖ⟩, which is cheap to compute: apply the corresponding Pauli (X, Y, or Z) to |ψₖ⟩ and multiply by −i/2.

### The backward pass

The key insight is that ⟨λ| can be **propagated backwards** gate by gate at unit cost. Starting from:

```
⟨λₙ| = ⟨ψ| H    (= ket: |λₙ⟩ = H|ψ⟩)
```

We peel off gates from the right one at a time:

```
⟨λₖ₋₁| = ⟨λₖ| Uₖ   ⟺   |λₖ₋₁⟩ = Uₖ† |λₖ⟩
```

Simultaneously, we propagate |ψ⟩ backwards:

```
|ψₖ₋₁⟩ = Uₖ† |ψₖ⟩
```

At each parameterized gate we compute ∂⟨H⟩/∂θₖ = 2 Re⟨λₖ | (−i/2 Pₙ Uₖ) |ψₖ₋₁⟩ before unpeeling.

The full algorithm in pseudocode:

```
|ψ⟩ ← forward(circuit)        # one pass, O(2ⁿ) memory
|λ⟩ ← H|ψ⟩                   # Hamiltonian application

for k = n, n-1, …, 1:
    if gate k is parameterized:
        |δ⟩ ← (−i/2 Pₙ) Uₖ |ψ⟩    # derivative state
        grad[k] ← 2 Re⟨λ|δ⟩

    |ψ⟩ ← Uₖ† |ψ⟩
    |λ⟩ ← Uₖ† |λ⟩
```

**Cost:** 1 forward + 1 backward = 2 statevector passes, fixed, regardless of how many parameters.

### Memory

Both methods keep at most a constant number of statevectors in memory at once (here: |ψ⟩, |λ⟩, |δ⟩). Parameter-shift also uses O(1) statevectors but requires 2p full circuit evaluations. There is no O(p) or O(n) statevector overhead in either method — the difference is purely in wall-clock time.

---

## Concrete comparison on MetalQ

A 12-qubit VQE ansatz with 48 parameters (a representative number for a hardware-efficient ansatz of depth 4):

| Method | Circuit executions | Gradient wall time (est.) |
|--------|-------------------:|:--------------------------|
| Parameter-shift | 96 | 96 × 3.3 ms ≈ 317 ms |
| Adjoint | 2 | ≈ 6–8 ms |

The adjoint method is roughly **40× faster** on this circuit at 12 qubits. The gap widens with qubit count because each circuit execution is more expensive but the ratio of 2 vs 2p evaluations stays fixed.

---

## What this means for the VQE / QAOA inner loop

Variational algorithms are gradient-descent loops. Each step needs a gradient. With parameter-shift, a VQE with 1000 gradient steps and 48 parameters costs 96 000 circuit evaluations. With adjoint it is 2 000. That is a 48× reduction in simulation time, which means the difference between an experiment that runs in minutes and one that runs for hours.

This is why every serious statevector simulator (Qiskit Aer's `statevector` method, PennyLane's `default.qubit`, Pennylane Lightning) uses adjoint differentiation internally and reserves parameter-shift for hardware backends where you have no choice.

---

## Implementation notes

In `metal_quantum_native/src/gradient.rs`:

- The forward pass runs on the Metal GPU via the standard gate dispatch path.
- The backward pass (`apply_gate_dagger_cpu`) runs on the CPU because the backward loop has sequential data dependencies that make GPU parallelism across gates ineffective.
- The derivative state is computed by `apply_gate_derivative_cpu`, which applies the corresponding Pauli then multiplies by −i/2 (implemented as a swap of real/imaginary parts with sign changes: multiply by −i means (re, im) → (im, −re), then divide by 2).
- The inner product ⟨λ|δ⟩ is a simple dot product over the 2ⁿ complex amplitudes.

---

## References

- Jones, T. & Gacon, J. (2020). *Efficient calculation of gradients in classical simulations of variational quantum algorithms.* arXiv:2009.02823.
- Nielsen, M. A. & Chuang, I. L. (2000). *Quantum Computation and Quantum Information.* Cambridge University Press. (Chapter 4 for gate algebra, Chapter 8 for the generator decomposition.)
- Schuld, M., Bergholm, V., Gogolin, C., Izaac, J., & Killoran, N. (2019). *Evaluating analytic gradients on quantum hardware.* Physical Review A, 99(3).
