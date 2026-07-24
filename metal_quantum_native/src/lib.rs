mod gates;
mod simulator;
mod metal_backend;
mod gradient;
mod sampler;
mod hamiltonian;

use simulator::Simulator;
use std::ffi::c_void;

#[repr(C)]
pub struct GateParam {
    pub gate_id: u32,
    pub target:  u32,
    pub control: i32,
    pub theta:   f32,
    pub phi:     f32,
    pub lam:     f32,
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn mq_create(n_qubits: u32) -> *mut c_void {
    let sim = Box::new(Simulator::new(n_qubits as usize));
    Box::into_raw(sim) as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn mq_destroy(ptr: *mut c_void) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr as *mut Simulator));
    }
}

#[no_mangle]
pub unsafe extern "C" fn mq_reset(ptr: *mut c_void) {
    sim_mut(ptr).reset();
}

// ── Gate application ──────────────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn mq_apply_gate(
    ptr:     *mut c_void,
    gate_id: u32,
    target:  u32,
    control: i32,
    theta:   f32,
    phi:     f32,
    lam:     f32,
) {
    sim_mut(ptr).apply_gate(gate_id, target as usize, control, theta, phi, lam);
}

/// Batched gate application — encodes all gates in one GPU command buffer.
/// 100×+ faster than per-gate mq_apply_gate for deep circuits.
#[no_mangle]
pub unsafe extern "C" fn mq_apply_gates(
    ptr:    *mut c_void,
    gates:  *const GateParam,
    n_gates: u32,
) {
    let gates_slice = std::slice::from_raw_parts(gates, n_gates as usize);
    sim_mut(ptr).apply_gates(gates_slice);
}

// ── State access ──────────────────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn mq_statevector(
    ptr: *mut c_void,
    out: *mut f32,
    len: u32,
) {
    let sv = sim_ref(ptr).statevector();
    let n  = (len as usize / 2).min(sv.len());
    for i in 0..n {
        *out.add(i * 2)     = sv[i].re;
        *out.add(i * 2 + 1) = sv[i].im;
    }
}

/// Debug escape hatch: write an interleaved (re,im) f32 buffer into the
/// statevector and renormalize.  NOT part of the final HHL algorithm —
/// used only to bypass state prep while debugging QPE.
#[no_mangle]
pub unsafe extern "C" fn mq_set_state(ptr: *mut c_void, data: *const f32, len: u32) {
    sim_mut(ptr).set_state(data, len);
}

// ── Measurement & expectation ─────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn mq_sample(
    ptr:   *mut c_void,
    shots: u32,
    out:   *mut u64,
) -> u32 {
    let samples = crate::sampler::sample(sim_ref(ptr), shots as usize);
    let n = samples.len() as u32;
    for (i, s) in samples.iter().enumerate() {
        *out.add(i) = *s;
    }
    n
}

#[no_mangle]
pub unsafe extern "C" fn mq_expect(
    ptr:               *mut c_void,
    pauli_ids:         *const u8,
    qubits:            *const u32,
    coeffs:            *const f64,
    n_terms:           u32,
    n_qubits_per_term: u32,
) -> f64 {
    hamiltonian::expect(
        sim_ref(ptr), pauli_ids, qubits, coeffs,
        n_terms as usize, n_qubits_per_term as usize,
    )
}

/// Fixed spelling: was mq_adjoin_gradient (missing 't').
#[no_mangle]
pub unsafe extern "C" fn mq_adjoint_gradient(
    ptr:        *mut c_void,
    pauli_ids:  *const u8,
    qubits:     *const u32,
    coeffs:     *const f64,
    n_terms:    u32,
    npqt:       u32,
    gate_params: *const GateParam,
    n_gates:    u32,
    param_mask: *const u8,
    grad_out:   *mut f64,
) {
    gradient::adjoint(
        sim_ref(ptr), pauli_ids, qubits, coeffs, n_terms as usize, npqt as usize,
        std::slice::from_raw_parts(gate_params, n_gates as usize),
        std::slice::from_raw_parts(param_mask, n_gates as usize),
        std::slice::from_raw_parts_mut(grad_out, n_gates as usize),
    )
}

// ── Internal helpers ──────────────────────────────────────────────────────────

unsafe fn sim_mut<'a>(ptr: *mut c_void) -> &'a mut Simulator {
    &mut *(ptr as *mut Simulator)
}
unsafe fn sim_ref<'a>(ptr: *mut c_void) -> &'a Simulator {
    &*(ptr as *mut Simulator)
}