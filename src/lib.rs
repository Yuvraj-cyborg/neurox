//! # This is a numerical computation crate for Rust
//!
//! This crate (v0.1.0) provides core building blocks for numerical computing and machine learning,
//! including a flexible `Tensor` type, essential mathematical operations, and a `Model` abstraction
//! suitable for implementing feed-forward networks and beyond.
//!
//! Designed for extendability, it serves as a foundation for experimenting with ML models,
//! custom layers, and future GPU acceleration support.

pub mod model;
pub mod ops;
pub mod tensor;

pub use model::Model;
pub use tensor::Tensor;
