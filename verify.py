"""
verify.py — Smoke test for the MetalQ stack.
Runs four checks against the Rust/Metal backend through the Python FFI layer.
"""

import numpy as np
from metalq import Circuit, run, statevector, expect, Z
from metalq import Parameter


def test_bell_state():
    print("Test 1: Bell State")
    qc = Circuit(2)
    qc.h(0)
    qc.cx(0, 1)

    result = run(qc, shots=1000)
    print(f"  Counts: {result.counts}")

    # A Bell state should only produce 00 and 11, roughly 50/50
    assert '00' in result.counts, "Missing |00⟩ outcome"
    assert '11' in result.counts, "Missing |11⟩ outcome"
    # Cross terms should be absent (or vanishingly rare)
    assert result.counts.get('01', 0) < 20, "Unexpected |01⟩ leakage"
    assert result.counts.get('10', 0) < 20, "Unexpected |10⟩ leakage"
    print("  PASSED\n")


def test_statevector():
    print("Test 2: Statevector of H|0⟩")
    qc = Circuit(1)
    qc.h(0)

    sv = statevector(qc)
    print(f"  |ψ⟩ = {sv.vector}")

    expected = 1.0 / np.sqrt(2)
    assert abs(sv.vector[0].real - expected) < 1e-4, "Amplitude[0] wrong"
    assert abs(sv.vector[1].real - expected) < 1e-4, "Amplitude[1] wrong"
    print("  PASSED\n")


def test_parameterized_gate():
    print("Test 3: Parameterized RX(π) gate")
    qc = Circuit(1)
    theta = Parameter('theta')
    qc.rx(theta, 0)

    bound = qc.bind_parameters({'theta': np.pi})
    sv = statevector(bound)
    print(f"  |ψ⟩ = {sv.vector}")

    # RX(π)|0⟩ = -i|1⟩, so all probability should be on |1⟩
    probs = sv.probabilities()
    assert abs(probs[1] - 1.0) < 1e-3, f"Expected P(|1⟩)≈1, got {probs[1]}"
    print("  PASSED\n")


def test_expectation():
    print("Test 4: Expectation <Z> on |0⟩")
    qc = Circuit(1)
    h = Z(0)

    val = expect(qc, h)
    print(f"  <Z> = {val}")

    # |0⟩ is the +1 eigenstate of Z
    assert abs(val - 1.0) < 1e-4, f"Expected <Z>=1, got {val}"
    print("  PASSED\n")


def test_expectation_excited():
    print("Test 5: Expectation <Z> on X|0⟩ = |1⟩")
    qc = Circuit(1)
    qc.x(0)
    h = Z(0)

    val = expect(qc, h)
    print(f"  <Z> = {val}")

    # |1⟩ is the -1 eigenstate of Z
    assert abs(val - (-1.0)) < 1e-4, f"Expected <Z>=-1, got {val}"
    print("  PASSED\n")


if __name__ == "__main__":
    print("=" * 50)
    print("MetalQ Verification Suite")
    print("=" * 50 + "\n")

    try:
        test_bell_state()
        test_statevector()
        test_parameterized_gate()
        test_expectation()
        test_expectation_excited()

        print("=" * 50)
        print("All tests passed ✓")
        print("=" * 50)
    except AssertionError as e:
        print(f"\n  ✗ ASSERTION FAILED: {e}")
        print("\n  The backend ran but produced wrong values.")
        print("  Check the gate math in your .metal shaders.")
    except Exception as e:
        print(f"\n  ✗ ERROR: {type(e).__name__}: {e}")
        print("\n  Common causes:")
        print("   - dylib not found → check target/release/libmetalq_native.dylib exists")
        print("   - 'Failed to load Metal library' → MetalKernels.metallib path mismatch")
        print("   - segfault → FFI signature mismatch between _ffi.py and lib.rs")
        import traceback
        traceback.print_exc()