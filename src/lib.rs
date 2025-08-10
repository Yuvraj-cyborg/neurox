//! Neurox — minimalist numerical and ML building blocks in Rust.
//!
//! This crate provides:
//! - A `Tensor` type for numeric data (row-major)
//! - Core tensor ops (matrix multiplication, element-wise add/mul, broadcasting)
//! - Activation functions (ReLU, Sigmoid, Tanh, Softmax)
//! - A `Dense` layer with backprop and cached activations
//! - A sequential `Model` with forward pass and training via SGD/Adam
//! - Losses (MSE, Cross-Entropy), data utilities, and error types
//!
//! All operations currently run on CPU and are designed for clarity and extensibility.
//!
//! Example: small MLP with ReLU
//! ```ignore
//! use neurox::{Model, Tensor, Activation};
//!
//! // 3 -> 4 -> 2 MLP
//! let mut model = Model::new(&[3, 4, 2], Activation::ReLU);
//!
//! // batch of 8 samples with 3 features
//! let x = Tensor::random(8, 3);
//! let logits = model.forward(&x).unwrap();
//! let probs = neurox::activations::softmax(&logits);
//! println!("probs: {:?}", probs);
//! ```

pub mod activations;
pub mod data;
pub mod errors;
pub mod layers;
pub mod loss;
pub mod model;
pub mod optimizer;
pub mod ops;
pub mod tensor;
pub mod utils;

// Convenient re-exports for common types and errors
pub use crate::{model::Model, tensor::Tensor};
pub use crate::layers::{Dense, Activation};
pub use crate::optimizer::{SGD, Adam};
pub use crate::errors::{NeuroxError, NeuroxResult};

/// Prelude with the most commonly used items.
pub mod prelude {
    pub use crate::{Tensor, Model};
    pub use crate::layers::{Dense, Activation};
    pub use crate::optimizer::{SGD, Adam};
    pub use crate::errors::{NeuroxError, NeuroxResult};
}
