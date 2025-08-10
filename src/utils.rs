use rand::{SeedableRng, rngs::StdRng};

/// Set global seed for reproducibility (affects rand::thread_rng only if used indirectly).
pub fn set_seed(seed: u64) {
    let _rng = StdRng::seed_from_u64(seed);
    println!("Seed set to {}", seed);
}
