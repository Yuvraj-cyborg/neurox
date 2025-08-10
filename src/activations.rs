//! Provides activation functions and their derivatives for neural networks.

use crate::tensor::Tensor;

/// Applies the Rectified Linear Unit (ReLU) activation function element-wise.
///
/// The function is defined as $f(x) = \max(0, x)$.
pub fn relu(x: &Tensor) -> Tensor {
    x.map(|v| if v > 0.0 { v } else { 0.0 })
}

/// Computes the gradient of the ReLU function.
///
/// The derivative is $f'(x) = 1$ if $x > 0$, and $0$ otherwise.
pub fn relu_grad(x: &Tensor) -> Tensor {
    x.map(|v| if v > 0.0 { 1.0 } else { 0.0 })
}

/// Applies the Sigmoid activation function element-wise.
///
/// The function is defined as $\sigma(x) = \frac{1}{1 + e^{-x}}$.
pub fn sigmoid(x: &Tensor) -> Tensor {
    x.map(|v| 1.0 / (1.0 + (-v).exp()))
}

/// Computes the gradient of the Sigmoid function from its output.
///
/// This is an optimization that uses the output of the sigmoid, $s = \sigma(x)$,
/// to calculate the gradient as $\sigma'(x) = s \cdot (1 - s)$.
pub fn sigmoid_grad_from_out(sig_out: &Tensor) -> Tensor {
    sig_out.map(|s| s * (1.0 - s))
}

/// Applies the Hyperbolic Tangent (tanh) activation function element-wise.
///
/// The function is defined as $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$.
pub fn tanh(x: &Tensor) -> Tensor {
    x.map(|v| v.tanh())
}

/// Computes the gradient of the tanh function from its output.
///
/// This optimization uses the output of tanh, $t = \tanh(x)$, to calculate
/// the gradient as $\tanh'(x) = 1 - t^2$.
pub fn tanh_grad_from_out(tanh_out: &Tensor) -> Tensor {
    tanh_out.map(|t| 1.0 - t * t)
}

/// Applies the Softmax function to each row of the input tensor.
///
/// This implementation is numerically stable, preventing overflow by subtracting
/// the maximum value in each row before exponentiation. The function is defined as:
/// $$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$$
pub fn softmax(x: &Tensor) -> Tensor {
    let mut out = x.clone();
    for i in 0..x.rows {
        // Find max in row for numerical stability
        let mut max = f32::NEG_INFINITY;
        for j in 0..x.cols {
            max = max.max(x.get(i, j));
        }

        // Exponentiate and sum
        let mut sum = 0.0;
        for j in 0..x.cols {
            let v = (x.get(i, j) - max).exp();
            out.set(i, j, v);
            sum += v;
        }

        // Normalize
        for j in 0..x.cols {
            out.set(i, j, out.get(i, j) / sum);
        }
    }
    out
}
