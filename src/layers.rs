//! Defines the layers of a neural network, such as the `Dense` layer.

use crate::errors::NeuroxResult;
use crate::{activations, ops, tensor::Tensor};

/// A fully-connected (dense) neural network layer.
///
/// A dense layer applies a linear transformation $Y = XW + B$ followed by an
/// optional activation function. It stores caches from the forward pass
/// which are required for backpropagation.
pub struct Dense {
    /// Weight matrix of shape `(in_features, out_features)`.
    pub w: Tensor,
    /// Bias vector of shape `(1, out_features)`.
    pub b: Tensor,
    /// The activation function to apply after the linear transformation.
    pub activation: Activation,

    // Caches for backpropagation
    input_cache: Option<Tensor>,
    preact_cache: Option<Tensor>,

    /// Gradient of the loss with respect to the weights, computed during the backward pass.
    pub grad_w: Option<Tensor>,
    /// Gradient of the loss with respect to the biases, computed during the backward pass.
    pub grad_b: Option<Tensor>,
}

/// An enumeration of supported activation functions for a layer.
#[derive(Clone, Copy, Debug)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    None,
}

impl Dense {
    /// Creates a new `Dense` layer with random weights and biases.
    ///
    /// # Arguments
    ///
    /// * `in_features` - The number of input features (columns of the input tensor).
    /// * `out_features` - The number of output features (columns of the output tensor).
    /// * `activation` - The `Activation` function to use for this layer.
    pub fn new(in_features: usize, out_features: usize, activation: Activation) -> Self {
        Dense {
            w: Tensor::random(in_features, out_features),
            b: Tensor::random(1, out_features),
            input_cache: None,
            preact_cache: None,
            grad_w: None,
            grad_b: None,
            activation,
        }
    }

    /// Performs the forward pass for the layer.
    ///
    /// Computes `activation(input @ w + b)`. The input and pre-activation
    /// tensors are cached for use in the backward pass.
    ///
    /// # Arguments
    /// * `input` - A tensor of shape `(batch_size, in_features)`.
    ///
    /// # Returns
    /// A `Result` containing the output tensor of shape `(batch_size, out_features)`,
    /// or an error if dimensions are mismatched.
    pub fn forward(&mut self, input: &Tensor) -> NeuroxResult<Tensor> {
        self.input_cache = Some(input.clone());

        let z = ops::matmul(input, &self.w)?;
        let z = z.add_row_broadcast(&self.b)?;
        self.preact_cache = Some(z.clone());

        let out = match self.activation {
            Activation::ReLU => activations::relu(&z),
            Activation::Sigmoid => activations::sigmoid(&z),
            Activation::Tanh => activations::tanh(&z),
            Activation::None => z,
        };
        Ok(out)
    }

    /// Performs the backward pass (backpropagation) for the layer.
    ///
    /// Given the gradient of the loss with respect to the layer's output (`grad_out`),
    /// this method computes the gradient of the loss with respect to the layer's
    /// parameters (`grad_w`, `grad_b`) and input (`grad_input`).
    ///
    /// # Panics
    ///
    /// Panics if `forward()` was not called before `backward()`.
    ///
    /// # Arguments
    /// * `grad_out` - The gradient from the subsequent layer, with shape `(batch_size, out_features)`.
    ///
    /// # Returns
    /// The gradient with respect to this layer's input (`dL/dX`), with shape `(batch_size, in_features)`.
    pub fn backward(&mut self, grad_out: &Tensor) -> NeuroxResult<Tensor> {
        let pre = self
            .preact_cache
            .as_ref()
            .expect("forward pass must be called before backward");

        // Gradient of the loss w.r.t. pre-activation (dL/dZ) using the chain rule.
        // dL/dZ = dL/dOut * dOut/dZ (element-wise)
        let dz = match self.activation {
            Activation::ReLU => {
                let g = activations::relu_grad(pre);
                crate::ops::mul_elementwise(grad_out, &g)?
            }
            Activation::Sigmoid => {
                let out = activations::sigmoid(pre);
                let g = activations::sigmoid_grad_from_out(&out);
                crate::ops::mul_elementwise(grad_out, &g)?
            }
            Activation::Tanh => {
                let out = activations::tanh(pre);
                let g = activations::tanh_grad_from_out(&out);
                crate::ops::mul_elementwise(grad_out, &g)?
            }
            Activation::None => grad_out.clone(),
        };

        // Gradient for weights (dL/dW) = X^T * dL/dZ
        let input = self.input_cache.as_ref().expect("no input cache");
        let gw = ops::matmul(&input.transpose(), &dz)?;

        // Gradient for biases (dL/dB) = sum of dL/dZ rows
        let mut gb = Tensor::zeros(1, dz.cols);
        for j in 0..dz.cols {
            let mut s = 0.0;
            for i in 0..dz.rows {
                s += dz.get(i, j);
            }
            gb.set(0, j, s);
        }

        // Gradient to pass to the previous layer (dL/dX) = dL/dZ * W^T
        let grad_input = ops::matmul(&dz, &self.w.transpose())?;

        self.grad_w = Some(gw);
        self.grad_b = Some(gb);

        Ok(grad_input)
    }

    /// Updates the layer's weights and biases using the stored gradients.
    ///
    /// This performs a single step of Stochastic Gradient Descent (SGD):
    /// `param = param - learning_rate * grad_param`.
    pub fn apply_gradients(&mut self, lr: f32) {
        if let Some(gw) = &self.grad_w {
            for idx in 0..self.w.data.len() {
                self.w.data[idx] -= lr * gw.data[idx];
            }
        }
        if let Some(gb) = &self.grad_b {
            for idx in 0..self.b.data.len() {
                self.b.data[idx] -= lr * gb.data[idx];
            }
        }
    }

    /// Returns the total number of trainable parameters in the layer (weights and biases).
    pub fn num_params(&self) -> usize {
        self.w.data.len() + self.b.data.len()
    }
}
