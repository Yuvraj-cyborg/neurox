# Neurox

[<img alt="github" src="https://img.shields.io/badge/github-yuvrajbiswal/neurox-8da0cb?style=for-the-badge&labelColor=555555&logo=github" height="20">](https://github.com/yuvrajbiswal/neurox)
[<img alt="crates.io" src="https://img.shields.io/crates/v/neurox.svg?style=for-the-badge&color=fc8d62&logo=rust" height="20">](https://crates.io/crates/neurox)
[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-neurox-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs" height="20">](https://docs.rs/neurox)


---

## Overview

**Neurox** is a fast, minimalist, and extendable numerical computation & machine learning library written in **Rust**.  
It provides **tensor operations**, **activation functions**, **layer abstractions**, and **model building blocks** to create and run ML models.  
Currently optimized for CPU execution, with a GPU backend planned in future releases.

**Perfect for:**
- Learning how ML frameworks work under the hood
- Building lightweight ML models in Rust
- Using as a base for larger GPU-accelerated projects

---

## Features (v0.2.0)

- **Multi-dimensional Tensor** struct for efficient numerical storage
- **Matrix operations**: multiplication, addition, dot products
- **Activation functions**: ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, XOR-like logical ops
- **Layer system**: Dense layers with bias & activation support
- **Model API**: Create, add layers, run forward passes
- **Logical / Boolean operations on tensors** (e.g., XOR, AND, OR)
- **Random initialization utilities**
- **Device abstraction** for future GPU acceleration
- **Modular architecture** extend with custom layers or activations easily
- **Example scripts** for quick usage

---

## 📦 Installation

Add **Neurox** to your Rust project:

```toml
[dependencies]
neurox = "0.2.0"
