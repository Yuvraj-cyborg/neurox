//! Provides basic mathematical operations for `Tensor`s.

use crate::{errors::NeuroxError, errors::NeuroxResult, tensor::Tensor};

/// Performs matrix multiplication on two tensors, `a` and `b`.
///
/// Calculates $C = A \times B$, where `a` has shape `(m, k)` and `b` has shape `(k, n)`.
/// The resulting tensor `C` will have shape `(m, n)`.
///
/// # Errors
///
/// Returns `NeuroxError::ShapeMismatch` if `a.cols` is not equal to `b.rows`.
pub fn matmul(a: &Tensor, b: &Tensor) -> NeuroxResult<Tensor> {
    if a.cols != b.rows {
        return Err(NeuroxError::ShapeMismatch(
            "a.cols must equal b.rows for matmul".into(),
        ));
    }
    let m = a.rows;
    let k = a.cols;
    let n = b.cols;
    let mut out = Tensor::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for t in 0..k {
                s += a.get(i, t) * b.get(t, j);
            }
            out.set(i, j, s);
        }
    }
    Ok(out)
}

/// Performs element-wise addition of two tensors.
///
/// # Errors
///
/// Returns `NeuroxError::ShapeMismatch` if `a` and `b` do not have identical shapes.
pub fn add(a: &Tensor, b: &Tensor) -> NeuroxResult<Tensor> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(NeuroxError::ShapeMismatch(
            "tensors must have the same shape for element-wise add".into(),
        ));
    }
    let data = a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect();
    Ok(Tensor::from_data(data, a.rows, a.cols))
}

/// Performs element-wise multiplication of two tensors.
///
/// # Errors
///
/// Returns `NeuroxError::ShapeMismatch` if `a` and `b` do not have identical shapes.
pub fn mul_elementwise(a: &Tensor, b: &Tensor) -> NeuroxResult<Tensor> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(NeuroxError::ShapeMismatch(
            "tensors must have the same shape for element-wise mul".into(),
        ));
    }
    let data = a.data.iter().zip(&b.data).map(|(x, y)| x * y).collect();
    Ok(Tensor::from_data(data, a.rows, a.cols))
}
