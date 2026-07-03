use metal::*;
use num_complex::Complex;
use crate::gates;

pub struct MetalBackend {
    device: Device,
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
    pub fn new() -> Self {
        let device = Device::system_default().expect("No Metal device found");
        eprintln!("[MetalQ] GPU backend: {}", device.name());
        Self { device }
    }

    pub fn apply_gate(
        &mut self,
        state:    &mut Vec<Complex<f32>>,
        _n_qubits: usize,
        gate_id:  u32,
        target:   usize,
        control:  i32,
        theta:    f32,
        phi:      f32,
        lam:      f32,
    ) {
        let library = self.device
            .new_library_with_file("MetalKernels.metallib")
            .expect("Failed to load Metal library");

        let kernel_name = gates::kernel_name(gate_id);
        let function = library.get_function(kernel_name, None).unwrap();

        let pipeline = self.device
            .new_compute_pipeline_state_with_function(&function)
            .unwrap();

        let cmd_queue  = self.device.new_command_queue();
        let cmd_buffer = cmd_queue.new_command_buffer();
        let encoder    = cmd_buffer.new_compute_command_encoder();

        // Use the real complex element size, not a hardcoded [f32; 2]
        let byte_len = (state.len() * std::mem::size_of::<Complex<f32>>()) as u64;
        let state_buf = self.device.new_buffer_with_data(
            state.as_ptr() as *const _,
            byte_len,
            MTLResourceOptions::StorageModeShared,
        );

        let params = GateParams { gate_id, target: target as u32, control, theta, phi, lam };
        let param_buf = self.device.new_buffer_with_data(
            &params as *const _ as *const _,
            std::mem::size_of::<GateParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&state_buf), 0);
        encoder.set_buffer(1, Some(&param_buf), 0);

        // Two-qubit gates touch 4 amplitudes per group → half the threads
        let n_threads = if gates::is_two_qubit(gate_id) {
            state.len() / 4
        } else {
            state.len() / 2
        };

        let grid = MTLSize::new(n_threads as u64, 1, 1);
        let tpt  = pipeline.max_total_threads_per_threadgroup();
        let threads = MTLSize::new(tpt.min(n_threads as u64).max(1), 1, 1);
        encoder.dispatch_threads(grid, threads);
        encoder.end_encoding();

        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        unsafe {
            let ptr = state_buf.contents() as *const Complex<f32>;
            let slice = std::slice::from_raw_parts(ptr, state.len());
            state.copy_from_slice(slice);
        }
    }
}