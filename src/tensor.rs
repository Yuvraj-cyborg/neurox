//! Defines the core `Tensor` struct and its associated methods.

use crate::errors::{NeuroxError, NeuroxResult};
use rand::Rng;
use std::fmt;

/// A 2D tensor representing a matrix of `f32` values, stored in row-major order.
#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Tensor {
    /// Creates a new tensor of `rows` x `cols` initialized with zeros.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Creates a new tensor from an existing data vector in row-major order.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` is not equal to `rows * cols`.
    pub fn from_data(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "Data size must match tensor dimensions."
        );
        Self { data, rows, cols }
    }

    /// Creates a new tensor with random values sampled from a uniform distribution between -1.0 and 1.0.
    pub fn random(rows: usize, cols: usize) -> Self {
        let mut rng = rand::rng();
        let data = (0..rows * cols)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        Self { data, rows, cols }
    }

    /// Returns the shape of the tensor as `(rows, cols)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Returns the value at the specified `(row, col)` index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }

    /// Sets the `value` at the specified `(row, col)` index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn set(&mut self, r: usize, c: usize, v: f32) {
        self.data[r * self.cols + c] = v;
    }

    /// Applies a function element-wise to the tensor, returning a new `Tensor`.
    pub fn map<F>(&self, mut f: F) -> Tensor
    where
        F: FnMut(f32) -> f32,
    {
        let d = self.data.iter().map(|&x| f(x)).collect();
        Tensor::from_data(d, self.rows, self.cols)
    }

    /// Adds a bias row vector to each row of this tensor (broadcasts).
    ///
    /// # Errors
    ///
    /// Returns `NeuroxError::ShapeMismatch` if the bias tensor's shape is not `(1, self.cols)`.
    pub fn add_row_broadcast(&self, bias: &Tensor) -> NeuroxResult<Tensor> {
        if bias.rows != 1 || bias.cols != self.cols {
            return Err(NeuroxError::ShapeMismatch(
                "bias shape must be (1, cols)".into(),
            ));
        }
        let mut out = self.clone();
        for i in 0..self.rows {
            for j in 0..self.cols {
                let idx = i * self.cols + j;
                out.data[idx] += bias.data[j];
            }
        }
        Ok(out)
    }

    /// Creates a new tensor of `rows` x `cols` initialized with zeros. Alias for `new`.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::new(rows, cols)
    }

    /// Returns a new `Tensor` that is the transpose of this one.
    pub fn transpose(&self) -> Tensor {
        let mut out = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                out[j * self.rows + i] = self.get(i, j);
            }
        }
        Tensor::from_data(out, self.cols, self.rows)
    }
}

/// Provides a truncated, pretty-printed format for debugging tensors.
impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = String::new();
        s.push_str(&format!("Tensor({}, {}) [", self.rows, self.cols));
        for i in 0..self.rows.min(3) {
            s.push('[');
            for j in 0..self.cols.min(6) {
                s.push_str(&format!("{:.4}", self.get(i, j)));
                if j + 1 < self.cols.min(6) {
                    s.push_str(", ");
                }
            }
            if self.cols > 6 {
                s.push_str(", ...");
            }
            s.push(']');
            if i + 1 < self.rows.min(3) {
                s.push_str(", ");
            }
        }
        if self.rows > 3 {
            s.push_str(", ...");
        }
        s.push(']');
        write!(f, "{}", s)
    }
}
