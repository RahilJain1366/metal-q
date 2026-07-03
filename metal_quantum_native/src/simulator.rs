use num_complex::Complex;
use crate::metal_backend::MetalBackend;

pub struct Simulator{
    pub n_qubits: usize,
    pub state: Vec<Complex<f32>>,
    backend: MetalBackend,
}

impl Simulator{
    pub fn new(n_qubits: usize) -> Self {
        let dim = 1 << n_qubits;
        let mut state = vec![Complex::new(0.0,0.0); dim];
        state[0] = Complex::new(1.0,0.0);
        Self {
            n_qubits, 
            state, 
            backend: MetalBackend::new(),
        }
    }

    pub fn reset(&mut self){
        self.state.fill(Complex::new(0.0,0.0));
        self.state[0] = Complex::new(1.0,0.0);
    }

    pub fn apply_gate(&mut self, gate_id: u32, target: usize, control: i32, thetha: f32, phi: f32, lam: f32){
        self.backend.apply_gate(&mut self.state, self.n_qubits, gate_id, target, control, thetha, phi, lam);
    }

    pub fn statevector(&self) -> &[Complex<f32>]{
        &self.state
    }
}
