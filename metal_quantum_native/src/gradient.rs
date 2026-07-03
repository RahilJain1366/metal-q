use num_complex::Complex;
use crate::simulator::Simulator;
use crate::GateParam;

/// Adjoint differentiation algorithm.
///
/// Forward pass:  run circuit, save final |ψ⟩
/// Backward pass: propagate |λ⟩ = H|ψ⟩ backward through gates,
///                at each parameterized gate compute:
///                grad[i] = 2 * Re⟨λ| dG/dθ |ψ⟩
pub unsafe fn adjoint(
    sim:        &Simulator,
    pauli_ids:  *const u8,
    qubits:     *const u32,
    coeffs:     *const f64,
    n_terms:    usize,
    n_qpt:      usize,
    gate_params: &[GateParam],
    param_mask:  &[u8],
    grad_out:    &mut [f64],
) {
    let n_qubits = sim.n_qubits;
    let dim      = 1usize << n_qubits;

    // ── Forward pass: build |ψ⟩ ──────────────────────────────────────────────
    let mut psi: Vec<Complex<f32>> = sim.statevector().to_vec();
    for gp in gate_params.iter() {
        apply_gate_cpu(&mut psi, n_qubits, gp);
    }

    // ── Initialize |λ⟩ = H|ψ⟩ ───────────────────────────────────────────────
    let mut lam = apply_hamiltonian_cpu(
        &psi, pauli_ids, qubits, coeffs, n_terms, n_qpt, dim,
    );

    // ── Backward pass ────────────────────────────────────────────────────────
    let mut grad_idx = grad_out.len();

    for (i, gp) in gate_params.iter().enumerate().rev() {
        if param_mask[i] == 1 {
            grad_idx -= 1;

            // Apply dG/dθ to |ψ⟩ — derivative of the gate w.r.t. theta
            let mut dpsi = psi.clone();
            apply_gate_derivative_cpu(&mut dpsi, n_qubits, gp);

            // grad = 2 * Re⟨λ|dG/dθ|ψ⟩
            let overlap: Complex<f64> = lam.iter()
                .zip(dpsi.iter())
                .map(|(l, d)| {
                    let lc = Complex::new(l.re as f64, l.im as f64);
                    let dc = Complex::new(d.re as f64, d.im as f64);
                    lc.conj() * dc
                })
                .sum();

            grad_out[grad_idx] = 2.0 * overlap.re;
        }

        // Unapply gate from |ψ⟩ (apply conjugate transpose G†)
        apply_gate_dagger_cpu(&mut psi, n_qubits, gp);

        // Unapply gate from |λ⟩
        apply_gate_dagger_cpu(&mut lam, n_qubits, gp);
    }
}

// ── Gate application on CPU (used during adjoint backward pass) ───────────────

fn apply_gate_cpu(
    state:    &mut Vec<Complex<f32>>,
    n_qubits: usize,
    gp:       &GateParam,
) {
    let n   = gp.target as usize;
    let dim = state.len();

    for gid in 0..(dim / 2) {
        let lo = (gid & !(1 << n)) | (0 << n);
        let hi = lo | (1 << n);

        // Control qubit check
        if gp.control >= 0 {
            let ctrl = gp.control as usize;
            if !((lo >> ctrl) & 1 == 1) { continue; }
        }

        let a = state[lo];
        let b = state[hi];
        let c = (gp.theta / 2.0).cos();
        let s = (gp.theta / 2.0).sin();

        let (na, nb) = match gp.gate_id {
            0 => { // H
                let f = 1.0 / 2.0f32.sqrt();
                (Complex::new((a.re+b.re)*f, (a.im+b.im)*f),
                 Complex::new((a.re-b.re)*f, (a.im-b.im)*f))
            }
            1 => (b, a), // X
            2 => { // Y
                (Complex::new( b.im, -b.re),
                 Complex::new(-a.im,  a.re))
            }
            3 => (a, Complex::new(-b.re, -b.im)), // Z
            4 => { // RX
                (Complex::new(c*a.re + s*b.im,  c*a.im - s*b.re),
                 Complex::new(s*a.im + c*b.re, -s*a.re + c*b.im))
            }
            5 => { // RY
                (Complex::new(c*a.re - s*b.re, c*a.im - s*b.im),
                 Complex::new(s*a.re + c*b.re, s*a.im + c*b.im))
            }
            6 => { // RZ
                let cr = (gp.theta/2.0).cos();
                let sr = (gp.theta/2.0).sin();
                (Complex::new(cr*a.re + sr*a.im, cr*a.im - sr*a.re),
                 Complex::new(cr*b.re - sr*b.im, cr*b.im + sr*b.re))
            }
            _ => (a, b)
        };

        state[lo] = na;
        state[hi] = nb;
    }
}

/// Applies G† (conjugate transpose) — used to unwind the forward pass
fn apply_gate_dagger_cpu(
    state:    &mut Vec<Complex<f32>>,
    n_qubits: usize,
    gp:       &GateParam,
) {
    // For unitary gates: G† = G with theta → -theta
    // H, X, Y, Z are self-adjoint (H† = H), so same gate
    let dagger_gp = GateParam {
        gate_id: gp.gate_id,
        target:  gp.target,
        control: gp.control,
        theta:   match gp.gate_id {
            0 | 1 | 2 | 3 => gp.theta,  // self-adjoint
            _ => -gp.theta,              // RX†=RX(-θ), RY†=RY(-θ), RZ†=RZ(-θ)
        },
        phi: -gp.phi,
        lam: -gp.lam,
    };
    apply_gate_cpu(state, n_qubits, &dagger_gp);
}

/// Applies dG/dθ — derivative of the gate with respect to theta
/// For rotation gates Rn(θ): dRn/dθ = -i/2 * Pn * Rn(θ)
/// In practice: dRx/dθ = -i/2 * X,  dRy/dθ = -i/2 * Y,  dRz/dθ = -i/2 * Z
fn apply_gate_derivative_cpu(
    state:    &mut Vec<Complex<f32>>,
    n_qubits: usize,
    gp:       &GateParam,
) {
    let derivative_gate_id = match gp.gate_id {
        4 => 1u32,  // dRX/dθ involves X
        5 => 2u32,  // dRY/dθ involves Y
        6 => 3u32,  // dRZ/dθ involves Z
        _ => return // non-parameterized gates have no theta derivative
    };

    // Apply the rotation gate first
    apply_gate_cpu(state, n_qubits, gp);

    // Then apply -i/2 * Pauli
    let pauli_gp = GateParam {
        gate_id: derivative_gate_id,
        target:  gp.target,
        control: gp.control,
        theta:   0.0,
        phi:     0.0,
        lam:     0.0,
    };
    apply_gate_cpu(state, n_qubits, &pauli_gp);

    // Multiply by -i/2 global factor
    for amp in state.iter_mut() {
        let re = amp.re;
        let im = amp.im;
        *amp = Complex::new(0.5 * im, -0.5 * re); // multiply by -i/2
    }
}

/// Applies H|ψ⟩ on CPU to initialize the backward lambda state
unsafe fn apply_hamiltonian_cpu(
    psi:       &[Complex<f32>],
    pauli_ids: *const u8,
    qubits:    *const u32,
    coeffs:    *const f64,
    n_terms:   usize,
    n_qpt:     usize,
    dim:       usize,
) -> Vec<Complex<f32>> {
    let mut result = vec![Complex::new(0.0f32, 0.0); dim];

    for t in 0..n_terms {
        let coeff = *coeffs.add(t) as f32;

        for i in 0..dim {
            let mut j     = i;
            let mut phase = Complex::new(1.0f32, 0.0);

            for k in 0..n_qpt {
                let pauli = *pauli_ids.add(t * n_qpt + k);
                let qubit = *qubits.add(t * n_qpt + k) as usize;

                match pauli {
                    0 => {}  // I
                    1 => { j ^= 1 << qubit; } // X
                    2 => {   // Y
                        let bit = (i >> qubit) & 1;
                        phase *= if bit == 0 {
                            Complex::new(0.0, 1.0)
                        } else {
                            Complex::new(0.0, -1.0)
                        };
                        j ^= 1 << qubit;
                    }
                    3 => {   // Z
                        if (i >> qubit) & 1 == 1 {
                            phase *= Complex::new(-1.0, 0.0);
                        }
                    }
                    _ => {}
                }
            }
            result[j] += Complex::new(
                coeff * (phase * psi[i]).re,
                coeff * (phase * psi[i]).im,
            );
        }
    }
    result
}