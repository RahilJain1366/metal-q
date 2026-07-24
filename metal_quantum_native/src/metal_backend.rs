use metal::*;
use num_complex::Complex;
use std::collections::HashMap;
use crate::gates;
use crate::GateParam;

pub struct MetalBackend {
    device:    Device,
    queue:     CommandQueue,
    pipelines: HashMap<&'static str, ComputePipelineState>,
    state_buf: Buffer,
    n_qubits:  usize,
}

#[repr(C)]
struct GateParams {
    gate_id: u32,
    target:  u32,
    control: i32,
    theta:   f32,
    phi:     f32,
    lam:     f32,
}

impl MetalBackend {
    pub fn new(n_qubits: usize) -> Self {
        let device = Device::system_default().expect("No Metal device found");
        eprintln!("[MetalQ] GPU: {}", device.name());

        // Load metallib embedded at compile time — no filesystem dependency.
        let metallib_bytes = include_bytes!("../../MetalKernels.metallib");
        let library = device
            .new_library_with_data(metallib_bytes)
            .expect("Failed to load embedded Metal library");

        let queue = device.new_command_queue();

        // Build all pipelines once up-front.
        let mut pipelines: HashMap<&'static str, ComputePipelineState> = HashMap::new();
        for &name in &["apply_gate", "apply_cnot", "apply_cz", "apply_swap"] {
            let function = library
                .get_function(name, None)
                .unwrap_or_else(|_| panic!("[MetalQ] Missing kernel: {name}"));
            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .expect("[MetalQ] Pipeline creation failed");
            pipelines.insert(name, pipeline);
        }

        // Persistent state buffer (StorageModeShared = CPU+GPU unified memory).
        let dim = 1usize << n_qubits;
        let byte_len = (dim * std::mem::size_of::<Complex<f32>>()) as u64;
        let state_buf = device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared);

        let mut backend = Self { device, queue, pipelines, state_buf, n_qubits };
        backend.reset();
        backend
    }

    // ── State management ──────────────────────────────────────────────────────

    pub fn reset(&mut self) {
        let dim = 1usize << self.n_qubits;
        unsafe {
            let ptr = self.state_buf.contents() as *mut Complex<f32>;
            std::slice::from_raw_parts_mut(ptr, dim).fill(Complex::new(0.0, 0.0));
            *ptr = Complex::new(1.0, 0.0);
        }
    }

    /// Returns the statevector by reading directly from the shared GPU buffer.
    /// No copy on Apple Silicon (unified memory).
    pub fn statevector(&self) -> &[Complex<f32>] {
        let dim = 1usize << self.n_qubits;
        unsafe {
            let ptr = self.state_buf.contents() as *const Complex<f32>;
            std::slice::from_raw_parts(ptr, dim)
        }
    }

    /// Writes an interleaved (re,im) f32 buffer into the statevector and renormalizes.
    /// Debug escape hatch — allows bypassing state prep to isolate QPE bugs.
    pub fn set_state(&mut self, data: *const f32, len: u32) {
        let dim = 1usize << self.n_qubits;
        let n_pairs = (len as usize / 2).min(dim);
        let raw = unsafe { std::slice::from_raw_parts(data, len as usize) };
        let mut norm_sq = 0.0f32;
        for i in 0..n_pairs {
            norm_sq += raw[i * 2] * raw[i * 2] + raw[i * 2 + 1] * raw[i * 2 + 1];
        }
        let norm = norm_sq.sqrt().max(1e-30);
        unsafe {
            let ptr = self.state_buf.contents() as *mut Complex<f32>;
            let slice = std::slice::from_raw_parts_mut(ptr, dim);
            for i in 0..n_pairs {
                slice[i] = Complex::new(raw[i * 2] / norm, raw[i * 2 + 1] / norm);
            }
            for i in n_pairs..dim {
                slice[i] = Complex::new(0.0, 0.0);
            }
        }
    }

    // ── Gate dispatch ─────────────────────────────────────────────────────────

    /// Apply a single gate.  Uses the cached pipeline and persistent state buffer.
    pub fn apply_gate(
        &mut self,
        gate_id: u32,
        target:  usize,
        control: i32,
        theta:   f32,
        phi:     f32,
        lam:     f32,
    ) {
        let gp = GateParam { gate_id, target: target as u32, control, theta, phi, lam };
        self.dispatch_gate(&gp);
    }

    /// Apply a slice of gates in one command buffer — one GPU submit for the whole batch.
    /// This amortises command-buffer overhead across many gates.
    pub fn apply_gates(&mut self, gates: &[GateParam]) {
        if gates.is_empty() { return; }
        let cmd_buf = self.queue.new_command_buffer();
        let dim = 1usize << self.n_qubits;
        for gp in gates {
            self.encode_gate(cmd_buf, gp, dim);
        }
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn dispatch_gate(&mut self, gp: &GateParam) {
        let cmd_buf = self.queue.new_command_buffer();
        let dim = 1usize << self.n_qubits;
        self.encode_gate(cmd_buf, gp, dim);
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
    }

    fn encode_gate(&self, cmd_buf: &CommandBufferRef, gp: &GateParam, dim: usize) {
        let params = GateParams {
            gate_id: gp.gate_id,
            target:  gp.target,
            control: gp.control,
            theta:   gp.theta,
            phi:     gp.phi,
            lam:     gp.lam,
        };
        let param_buf = self.device.new_buffer_with_data(
            &params as *const _ as *const _,
            std::mem::size_of::<GateParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let kernel_name = gates::kernel_name(gp.gate_id);
        let pipeline = &self.pipelines[kernel_name];
        let encoder = cmd_buf.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&self.state_buf), 0);
        encoder.set_buffer(1, Some(&param_buf), 0);

        let n_threads = if gates::is_two_qubit(gp.gate_id) { dim / 4 } else { dim / 2 };
        let tpt = pipeline.max_total_threads_per_threadgroup();
        let grid    = MTLSize::new(n_threads as u64, 1, 1);
        let threads = MTLSize::new(tpt.min(n_threads as u64).max(1), 1, 1);
        encoder.dispatch_threads(grid, threads);
        encoder.end_encoding();
    }
}