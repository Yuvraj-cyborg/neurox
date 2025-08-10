//! Provides mathematical operations for `Tensor`s.

use crate::tensor::Tensor;

/// Performs matrix multiplication on two tensors, `a` and `b`.
///
/// Calculates $C = A \times B$, where the resulting tensor `C` has dimensions
/// `(a.rows, b.cols)`.
///
/// # Panics
///
/// Panics if the number of columns in `a` is not equal to the number of rows in `b`.
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.cols, b.rows, "Matrix dimensions are incompatible for multiplication.");
    let mut result = Tensor::new(a.rows, b.cols);

    for i in 0..a.rows {
        for j in 0..b.cols {
            let mut sum = 0.0;
            for k in 0..a.cols {
                sum += a.get(i, k) * b.get(k, j);
            }
            result.set(i, j, sum);
        }
    }
    result
}

/// Applies the Rectified Linear Unit (ReLU) activation function element-wise.
///
/// The function is defined as $f(x) = \max(0, x)$.
pub fn relu(t: &Tensor) -> Tensor {
    let data: Vec<f32> = t.data.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect();
    Tensor::from_data(data, t.rows, t.cols)
}