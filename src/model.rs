//! Defines the neural network `Model` structure and its forward pass logic.

use crate::{ops, tensor::Tensor};

/// Represents a simple sequential feed-forward neural network.
///
/// The model is composed of a series of linear layers, each with its own
/// weight matrix and bias vector.
pub struct Model {
    weights: Vec<Tensor>,
    biases: Vec<Tensor>,
}

impl Model {
    /// Constructs a new `Model` from a specified architecture.
    ///
    /// The `layers` slice defines the number of neurons in each layer, starting
    /// with the input layer. For example, `&[784, 128, 10]` creates a network
    /// with an input size of 784, one hidden layer of 128 neurons, and an
    /// output layer of 10 neurons.
    ///
    /// Weights and biases are initialized with random values.
    pub fn new(layers: &[usize]) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for w in layers.windows(2) {
            weights.push(Tensor::random(w[0], w[1]));
            biases.push(Tensor::random(1, w[1]));
        }

        Self { weights, biases }
    }

    /// Performs a forward pass through the network.
    ///
    /// The input tensor is propagated through each layer, applying matrix
    /// multiplication with weights, adding biases, and using the ReLU
    /// activation function.
    pub fn forward(&self, mut input: Tensor) -> Tensor {
        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            input = ops::matmul(&input, w);
            input = add_bias(&input, b);
            input = ops::relu(&input);
        }
        input
    }
}

/// Adds a bias tensor `b` to each row of tensor `t`.
fn add_bias(t: &Tensor, b: &Tensor) -> Tensor {
    let mut result = t.clone();
    // Assuming b is a row vector (1, cols)
    for i in 0..t.rows {
        for j in 0..t.cols {
            result.set(i, j, result.get(i, j) + b.get(0, j));
        }
    }
    result
}