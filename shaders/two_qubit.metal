#include <metal_stdlib>
using namespace metal;

struct GateParams {
    uint  gate_id;
    uint  target;
    int   control;
    float theta;
    float phi;
    float lam;
};

// Insert a zero bit at position `bit` in `v`, shifting all higher bits up by 1.
static inline uint insert_zero(uint v, uint bit) {
    uint lo_mask = (1u << bit) - 1u;
    return ((v & ~lo_mask) << 1u) | (v & lo_mask);
}

// Build the base statevector index with both ctrl and tgt bits cleared.
// gid ∈ [0, N/4): each thread owns one 4-amplitude block.
static inline uint base_idx(uint gid, uint ctrl, uint tgt) {
    uint q0 = min(ctrl, tgt);
    uint q1 = max(ctrl, tgt);
    // Insert zeros at the two qubit positions (lower first so upper stays valid)
    return insert_zero(insert_zero(gid, q0), q1);
}

// CNOT: |c,t⟩ → |c, t⊕c⟩
// 4×4 (|00⟩,|01⟩,|10⟩,|11⟩): [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]
kernel void apply_cnot(
    device float2*       state  [[buffer(0)]],
    constant GateParams& params [[buffer(1)]],
    uint                 gid    [[thread_position_in_grid]])
{
    uint ctrl = (uint)params.control;
    uint tgt  = params.target;
    uint base = base_idx(gid, ctrl, tgt);

    // Only the |10⟩ ↔ |11⟩ pair is swapped (control bit = 1)
    uint i10 = base | (1u << ctrl);
    uint i11 = base | (1u << ctrl) | (1u << tgt);

    float2 tmp = state[i10];
    state[i10]  = state[i11];
    state[i11]  = tmp;
}

// CZ: phase-flip |11⟩ amplitude
// 4×4: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]]
kernel void apply_cz(
    device float2*       state  [[buffer(0)]],
    constant GateParams& params [[buffer(1)]],
    uint                 gid    [[thread_position_in_grid]])
{
    uint ctrl = (uint)params.control;
    uint tgt  = params.target;
    uint base = base_idx(gid, ctrl, tgt);

    uint i11 = base | (1u << ctrl) | (1u << tgt);
    state[i11] = -state[i11];
}

// SWAP: exchange |01⟩ and |10⟩ amplitudes
// 4×4: [[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]
kernel void apply_swap(
    device float2*       state  [[buffer(0)]],
    constant GateParams& params [[buffer(1)]],
    uint                 gid    [[thread_position_in_grid]])
{
    uint ctrl = (uint)params.control;
    uint tgt  = params.target;
    uint base = base_idx(gid, ctrl, tgt);

    // Each thread owns exactly one 4-amplitude block, so no double-swap guard needed.
    uint i01 = base | (1u << tgt);
    uint i10 = base | (1u << ctrl);

    float2 tmp = state[i01];
    state[i01]  = state[i10];
    state[i10]  = tmp;
}