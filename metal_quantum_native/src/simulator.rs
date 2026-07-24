use num_complex::Complex;
use crate::metal_backend::MetalBackend;
use crate::GateParam;

pub struct Simulator {
    pub n_qubits: usize,
    backend: MetalBackend,
}

impl Simulator {
    pub fn new(n_qubits: usize) -> Self {
        Self { n_qubits, backend: MetalBackend::new(n_qubits) }
    }

    pub fn reset(&mut self) {
        self.backend.reset();
    }

    pub fn apply_gate(
        &mut self, gate_id: u32, target: usize, control: i32,
        theta: f32, phi: f32, lam: f32,
    ) {
        self.backend.apply_gate(gate_id, target, control, theta, phi, lam);
    }

    pub fn apply_gates(&mut self, gates: &[GateParam]) {
        self.backend.apply_gates(gates);
    }

    pub fn statevector(&self) -> &[Complex<f32>] {
        self.backend.statevector()
    }

    pub fn set_state(&mut self, data: *const f32, len: u32) {
        self.backend.set_state(data, len);
    }
}