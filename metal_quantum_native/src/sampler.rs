use crate::simulator::Simulator;

pub fn sample(sim: &Simulator, shots: usize) -> Vec<u64> {
    let state = sim.statevector();
    
    // Build cumulative probability distribution
    let probs: Vec<f64> = state
        .iter()
        .map(|amp| (amp.re * amp.re + amp.im * amp.im) as f64)
        .collect();
    
    let mut cumulative = vec![0.0f64; probs.len()];
    cumulative[0] = probs[0];
    for i in 1..probs.len() {
        cumulative[i] = cumulative[i - 1] + probs[i];
    }

    // Draw shots using inverse CDF sampling
    let mut results = Vec::with_capacity(shots);
    for _ in 0..shots {
        let r = random_f64();
        // Binary search for the bucket this random value falls in
        let idx = cumulative
            .partition_point(|&p| p < r)
            .min(probs.len() - 1);
        results.push(idx as u64);
    }

    results
}

/// Simple xorshift64 — no external RNG dependency needed
fn random_f64() -> f64 {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u64> = Cell::new(
            // seed from current time
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos() as u64 | 1
        );
    }
    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        // Normalize to [0.0, 1.0)
        (x >> 11) as f64 / (1u64 << 53) as f64
    })
}