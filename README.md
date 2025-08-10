# Neurox

[![crates.io](https://img.shields.io/crates/v/neurox.svg)](https://crates.io/crates/neurox)
[![docs.rs](https://docs.rs/neurox/badge.svg)](https://docs.rs/neurox)
[![License](https://img.shields.io/crates/l/neurox.svg)](./LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70+-blue.svg)](https://www.rust-lang.org/)

---

## Overview

**Neurox** is a minimalist and extendable numerical computation library written in Rust.  
It provides core tensor operations, fundamental mathematical building blocks, and abstractions to build and experiment with machine learning models including but not limited to neural networks.  
All operations currently run on CPU, with plans for GPU acceleration support in future versions.

This crate is ideal for:
- Learning and experimenting with numerical computation primitives  
- Developing small-scale ML models from scratch in Rust  
- Serving as a foundation for more advanced GPU-accelerated ML frameworks

---

## Features (v0.1.0)

- Multi-dimensional `Tensor` struct for efficient numerical data storage  
- Core operations: matrix multiplication, element-wise activations (e.g., `ReLU`)  
- Flexible `Model` struct supporting layered architectures with bias  
- Forward pass computation suitable for inference and prototyping  
- Random weight initialization utilities  
- Device abstraction designed to enable future GPU support  

---

## Planned Features

- Training loops with backpropagation and optimizers (SGD, Adam)  
- Expanded activation functions (sigmoid, tanh, etc.)  
- GPU acceleration using CUDA, OpenCL, or WGPU  
- Additional layer types (Convolutional, Recurrent, etc.)  
- Serialization and model persistence utilities  
- Dataset handling and data loading features

---

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
neurox = "0.1.0"
