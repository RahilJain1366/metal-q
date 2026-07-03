use num_complex::Complex;
use crate::simulator::Simulator;

/// Pauli operator IDs — must match Python's PAULI_ID dict
const I: u8 = 0;
const X: u8 = 1;
const Y: u8 = 2;
const Z: u8 = 3;

/// Computes <ψ|H|ψ> for a PauliSum H.
///
/// pauli_ids: row-major (n_terms × n_qubits_per_term) array of Pauli IDs
/// qubits:    row-major (n_terms × n_qubits_per_term) array of qubit indices
/// coeffs:    length n_terms array of real coefficients
pub unsafe fn expect(
    sim:       &Simulator,
    pauli_ids: *const u8,
    qubits:    *const u32,
    coeffs:    *const f64,
    n_terms:   usize,
    n_qpt:     usize,   // n_qubits_per_term
) -> f64 {
    let state = sim.statevector();
    let dim   = state.len();
    let mut total = 0.0f64;

    for t in 0..n_terms {
        let coeff = *coeffs.add(t);
        // Compute <ψ|P_t|ψ> for this Pauli term
        let mut term_val = Complex::new(0.0f64, 0.0);

        for i in 0..dim {
            let mut j = i;           // index after applying Paulis
            let mut phase = Complex::new(1.0f64, 0.0);

            for k in 0..n_qpt {
                let pauli = *pauli_ids.add(t * n_qpt + k);
                let qubit = *qubits.add(t * n_qpt + k) as usize;

                match pauli {
                    I => { /* identity — no change */ }
                    X => {
                        // X flips qubit: |0⟩→|1⟩, |1⟩→|0⟩
                        j ^= 1 << qubit;
                    }
                    Y => {
                        // Y flips qubit and adds phase:
                        // Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
                        let bit = (i >> qubit) & 1;
                        phase *= if bit == 0 {
                            Complex::new(0.0, 1.0)   // i
                        } else {
                            Complex::new(0.0, -1.0)  // -i
                        };
                        j ^= 1 << qubit;
                    }
                    Z => {
                        // Z adds -1 phase if qubit is |1⟩
                        let bit = (i >> qubit) & 1;
                        if bit == 1 {
                            phase *= Complex::new(-1.0, 0.0);
                        }
                    }
                    _ => {}
                }
            }

            // <ψ|P|ψ> += conj(ψ[j]) * phase * ψ[i]
            let psi_i = Complex::new(state[i].re as f64, state[i].im as f64);
            let psi_j = Complex::new(state[j].re as f64, state[j].im as f64);
            term_val += psi_j.conj() * phase * psi_i;
        }

        total += coeff * term_val.re;
    }

    total
}