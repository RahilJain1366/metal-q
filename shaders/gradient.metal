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

// Re( conj(a) · b ) = a.x·b.x + a.y·b.y
static inline float re_inner(float2 a, float2 b) {
    return a.x * b.x + a.y * b.y;
}

// Adjoint gradient for rotation gates (RX=4, RY=5, RZ=6).
//
// Computes:  grad_buf[threadgroup_id] += 2·Re⟨λ| dG/dθ |ψ⟩  (partial per threadgroup)
// Rust must sum grad_buf[] to get the full scalar contribution for this gate.
//
// Dispatch: n_pairs = N/2 threads, threadgroup size must be a power-of-two.
// Buffer 3 must hold ceil(n_pairs / threadgroup_size) floats (zero-initialised).
// Rust must call:
//   encoder.set_threadgroup_memory_length(0, threadgroup_size * 4);
kernel void adjoint_gradient_step(
    device const float2*  psi        [[buffer(0)]],
    device const float2*  lam        [[buffer(1)]],
    constant GateParams&  params     [[buffer(2)]],
    device float*         grad_buf   [[buffer(3)]],
    uint                  gid        [[thread_position_in_grid]],
    uint                  lid        [[thread_position_in_threadgroup]],
    uint                  tg_id      [[threadgroup_position_in_grid]],
    uint                  tg_size    [[threads_per_threadgroup]],
    threadgroup float*    tg_partial [[threadgroup(0)]])
{
    uint n  = params.target;
    uint lo = ((gid >> n) << (n + 1u)) | (gid & ((1u << n) - 1u));
    uint hi = lo | (1u << n);

    float c    = cos(params.theta * 0.5f);
    float s    = sin(params.theta * 0.5f);
    float2 plo = psi[lo];
    float2 phi = psi[hi];
    float2 llo = lam[lo];
    float2 lhi = lam[hi];

    float partial = 0.0f;

    switch (params.gate_id) {

        case 4: {
            // dRX/dθ = (1/2)[[-s, -i·c], [-i·c, -s]]
            // (dRX/dθ)|ψ⟩[lo] = (1/2)(−s·plo − i·c·phi)
            //                  = (1/2)(−s·plo.x + c·phi.y,  −s·plo.y − c·phi.x)
            // (dRX/dθ)|ψ⟩[hi] = (1/2)(−i·c·plo − s·phi)
            //                  = (1/2)(c·plo.y − s·phi.x,  −c·plo.x − s·phi.y)
            float2 dlo = 0.5f * float2(-s*plo.x + c*phi.y, -s*plo.y - c*phi.x);
            float2 dhi = 0.5f * float2( c*plo.y - s*phi.x, -c*plo.x - s*phi.y);
            partial = re_inner(llo, dlo) + re_inner(lhi, dhi);
            break;
        }

        case 5: {
            // dRY/dθ = (1/2)[[-s, -c], [c, -s]]  (purely real)
            // (dRY/dθ)|ψ⟩[lo] = (1/2)(−s·plo − c·phi)
            // (dRY/dθ)|ψ⟩[hi] = (1/2)(c·plo − s·phi)
            float2 dlo = 0.5f * float2(-s*plo.x - c*phi.x, -s*plo.y - c*phi.y);
            float2 dhi = 0.5f * float2( c*plo.x - s*phi.x,  c*plo.y - s*phi.y);
            partial = re_inner(llo, dlo) + re_inner(lhi, dhi);
            break;
        }

        case 6: {
            // dRZ/dθ = (1/2)[[(−s−ic), 0], [0, (−s+ic)]]
            //   where (−s−ic)·plo = (−s·plo.x + c·plo.y,  −s·plo.y − c·plo.x)
            //         (−s+ic)·phi = (−s·phi.x − c·phi.y,  −s·phi.y + c·phi.x)
            float2 dlo = 0.5f * float2(-s*plo.x + c*plo.y, -s*plo.y - c*plo.x);
            float2 dhi = 0.5f * float2(-s*phi.x - c*phi.y, -s*phi.y + c*phi.x);
            partial = re_inner(llo, dlo) + re_inner(lhi, dhi);
            break;
        }
    }

    // Tree reduction within threadgroup → one atomic-free write per threadgroup
    tg_partial[lid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size >> 1u; stride > 0u; stride >>= 1u) {
        if (lid < stride) tg_partial[lid] += tg_partial[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) {
        grad_buf[tg_id] = tg_partial[0] * 2.0f;
    }
}