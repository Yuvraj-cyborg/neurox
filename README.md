# Neurox

[![crates.io](https://img.shields.io/crates/v/neurox.svg)](https://crates.io/crates/neurox)
[![docs.rs](https://docs.rs/neurox/badge.svg)](https://docs.rs/neurox)
[![License](https://img.shields.io/crates/l/neurox.svg)](./LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70+-blue.svg)](https://www.rust-lang.org/)

---

## Overview

**Neurox** is a minimal, extendable neural network library written in Rust, designed as a starting point for building and experimenting with deep learning models.  
It provides foundational tensor operations, simple feed-forward neural network layers, activation functions, and a basic training loop all CPU-based with GPU acceleration planned for future versions.

This crate is perfect for:
- Learning how neural networks work under the hood  
- Building small-scale neural models from scratch in Rust  
- Laying groundwork for advanced GPU-accelerated deep learning

---

## Features (v0.1.0)

- Multi-dimensional `Tensor` struct for numerical data storage  
- Core operations: matrix multiplication, element-wise activations (`ReLU`)  
- Simple `Model` struct supporting multiple dense layers with bias  
- Forward pass implementation for inference  
- Basic random weight initialization  
- Designed with a device abstraction for future GPU support  

---

## Planned Features

- Training loop with backpropagation and optimizers (SGD, Adam)  
- Support for more activation functions (sigmoid, tanh)  
- GPU acceleration using CUDA / WGPU  
- Additional layer types (Conv2D, LSTM)  
- Serialization / model saving and loading  
- Dataset utilities and data loaders

---

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
neurox = "0.1.0"
