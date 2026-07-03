pub const H:    u32 = 0;
pub const X:    u32 = 1;
pub const Y:    u32 = 2;
pub const Z:    u32 = 3;
pub const RX:   u32 = 4;
pub const RY:   u32 = 5;
pub const RZ:   u32 = 6;
pub const U:    u32 = 7;
pub const CNOT: u32 = 8;
pub const CZ:   u32 = 9;
pub const SWAP: u32 = 10;
pub const T:    u32 = 11;
pub const S:    u32 = 12;

/// True if this gate operates on two qubits (needs control + target indexing)
pub fn is_two_qubit(gate_id: u32) -> bool {
    matches!(gate_id, CNOT | CZ | SWAP)
}

/// Returns the Metal kernel function name for a given gate ID
pub fn kernel_name(gate_id: u32) -> &'static str {
    match gate_id {
        H | X | Y | Z | RX | RY | RZ | U | T | S => "apply_gate",
        CNOT => "apply_cnot",
        CZ   => "apply_cz",
        SWAP => "apply_swap",
        _    => panic!("Unknown gate id {gate_id}"),
    }
}