#include <metal_stdlib>
using namespace metal;

struct GateParams {
    uint  gate_id;
    uint  target;
    int   control;   // -1 = no control
    float theta;
    float phi;
    float lam;
};

kernel void apply_gate(
    device float2*       state  [[buffer(0)]],
    constant GateParams& params [[buffer(1)]],
    uint                 gid    [[thread_position_in_grid]])
{
    uint n = params.target;

    // Insert a zero bit at position n so that gid ∈ [0, N/2) maps to
    // the N statevector pairs without aliasing.
    // Bits above n shift left by 1; bits below n are unchanged.
    uint lo = ((gid >> n) << (n + 1u)) | (gid & ((1u << n) - 1u));
    uint hi = lo | (1u << n);

    // Controlled-gate guard: skip if control qubit is |0⟩
    if (params.control >= 0) {
        uint ctrl = (uint)params.control;
        if (!((lo >> ctrl) & 1u)) return;
    }

    float2 a = state[lo];
    float2 b = state[hi];

    // cos/sin of θ/2 used by rotation gates (unused for H/X/Y/Z/T/S, harmless)
    float c = cos(params.theta * 0.5f);
    float s = sin(params.theta * 0.5f);

    switch (params.gate_id) {

        case 0: {  // H: (1/√2) [[1, 1], [1, -1]]
            float inv_sqrt2 = 0.70710678118f;
            state[lo] = (a + b) * inv_sqrt2;
            state[hi] = (a - b) * inv_sqrt2;
            break;
        }

        case 1: {  // X: [[0, 1], [1, 0]]
            state[lo] = b;
            state[hi] = a;
            break;
        }

        case 2: {  // Y: [[0, -i], [i, 0]]
            state[lo] = float2( b.y, -b.x);   // -i·b
            state[hi] = float2(-a.y,  a.x);   //  i·a
            break;
        }

        case 3: {  // Z: [[1, 0], [0, -1]]
            state[hi] = float2(-b.x, -b.y);
            break;
        }

        case 4: {  // RX(θ): [[c, -i·s], [-i·s, c]]  (c=cos θ/2, s=sin θ/2)
            state[lo] = float2( c*a.x + s*b.y,  c*a.y - s*b.x);
            state[hi] = float2( c*b.x + s*a.y, -s*a.x + c*b.y);
            break;
        }

        case 5: {  // RY(θ): [[c, -s], [s, c]]  (purely real)
            state[lo] = float2(c*a.x - s*b.x,  c*a.y - s*b.y);
            state[hi] = float2(s*a.x + c*b.x,  s*a.y + c*b.y);
            break;
        }

        case 6: {  // RZ(θ): [[e^(-iθ/2), 0], [0, e^(iθ/2)]]
            state[lo] = float2( c*a.x + s*a.y,  c*a.y - s*a.x);
            state[hi] = float2( c*b.x - s*b.y,  c*b.y + s*b.x);
            break;
        }

        case 7: {  // U(θ,φ,λ): [[c, -e^(iλ)·s], [e^(iφ)·s, e^(i(φ+λ))·c]]
            float cp  = cos(params.phi);
            float sp  = sin(params.phi);
            float cl  = cos(params.lam);
            float sl  = sin(params.lam);
            float cpl = cos(params.phi + params.lam);
            float spl = sin(params.phi + params.lam);
            state[lo] = float2(
                c*a.x - s*(cl*b.x - sl*b.y),
                c*a.y - s*(cl*b.y + sl*b.x)
            );
            state[hi] = float2(
                s*(cp*a.x - sp*a.y) + c*(cpl*b.x - spl*b.y),
                s*(cp*a.y + sp*a.x) + c*(cpl*b.y + spl*b.x)
            );
            break;
        }

        case 11: {  // T: [[1, 0], [0, e^(iπ/4)]]
            // e^(iπ/4) = (1+i)/√2; cos(π/4)=sin(π/4)=1/√2
            float tcs = 0.70710678118f;
            state[hi] = float2(tcs*b.x - tcs*b.y,  tcs*b.y + tcs*b.x);
            break;
        }

        case 12: {  // S: [[1, 0], [0, i]]
            state[hi] = float2(-b.y, b.x);
            break;
        }
    }
}