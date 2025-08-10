//! Defines the main `Model` struct, its training loops, and evaluation utilities.

use crate::errors::NeuroxResult;
use crate::optimizer::{Adam, SGD};
use crate::{
    layers::{Activation, Dense},
    loss,
    tensor::Tensor,
};

/// A sequential feed-forward neural network model.
pub struct Model {
    pub layers: Vec<Dense>,
}

impl Model {
    /// Constructs a new `Model` from a specified architecture.
    ///
    /// # Arguments
    /// * `layer_sizes` - A slice defining the number of neurons in each layer,
    ///   e.g., `&[784, 128, 10]` for a 784-input, 128-hidden, 10-output network.
    /// * `activation` - The `Activation` function to use for all hidden layers.
    pub fn new(layer_sizes: &[usize], activation: Activation) -> Self {
        let mut layers = Vec::new();
        for win in layer_sizes.windows(2) {
            layers.push(Dense::new(win[0], win[1], activation));
        }
        Self { layers }
    }

    /// Performs a forward pass through the entire network.
    ///
    /// The output is the raw logits from the final layer, before any final
    /// activation like Softmax (which is typically handled by the loss function).
    ///
    /// # Returns
    /// A `Result` containing the output tensor (logits).
    pub fn forward(&mut self, input: &Tensor) -> NeuroxResult<Tensor> {
        let mut x = input.clone();
        for layer in self.layers.iter_mut() {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }

    /// Trains the model using the SGD optimizer.
    ///
    /// This method iterates through the dataset for a specified number of epochs,
    /// performing forward and backward passes and updating model weights.
    /// Assumes a Softmax Cross-Entropy loss for training.
    pub fn train_sgd(
        &mut self,
        x: &Tensor,
        y: &Tensor,
        epochs: usize,
        batch_size: usize,
        lr: f32,
    ) -> NeuroxResult<()> {
        let opt = SGD::new(lr);
        for _epoch in 0..epochs {
            // Naive batching without shuffling for simplicity.
            for start in (0..x.rows).step_by(batch_size) {
                let end = (start + batch_size).min(x.rows);
                let bx = slice_rows(x, start, end)?;
                let by = slice_rows(y, start, end)?;

                // Forward pass
                let preds = self.forward(&bx)?;
                // Assume Softmax Cross-Entropy loss
                let probs = crate::activations::softmax(&preds);
                let (_loss, grad) = loss::cross_entropy_loss(&probs, &by);

                // Backward pass through layers in reverse order
                let mut upstream_grad = grad;
                for layer in self.layers.iter_mut().rev() {
                    upstream_grad = layer.backward(&upstream_grad)?;
                }

                // Update weights
                opt.step(&mut self.layers);
            }
        }
        Ok(())
    }

    /// Trains the model using the Adam optimizer.
    ///
    /// This method iterates through the dataset for a specified number of epochs,
    /// performing forward and backward passes and updating model weights.
    /// Assumes a Softmax Cross-Entropy loss for training.
    pub fn train_adam(
        &mut self,
        x: &Tensor,
        y: &Tensor,
        epochs: usize,
        batch_size: usize,
        lr: f32,
    ) -> NeuroxResult<()> {
        let mut adam = Adam::new(lr, &self.layers);
        for _epoch in 0..epochs {
            for start in (0..x.rows).step_by(batch_size) {
                let end = (start + batch_size).min(x.rows);
                let bx = slice_rows(x, start, end)?;
                let by = slice_rows(y, start, end)?;

                let preds = self.forward(&bx)?;
                let probs = crate::activations::softmax(&preds);
                let (_loss, grad) = loss::cross_entropy_loss(&probs, &by);

                let mut upstream_grad = grad;
                for layer in self.layers.iter_mut().rev() {
                    upstream_grad = layer.backward(&upstream_grad)?;
                }

                adam.step(&mut self.layers);
            }
        }
        Ok(())
    }

    /// Prints a summary of the model's architecture and parameter counts.
    pub fn summary(&self) {
        println!("Model Summary:");
        let mut total = 0usize;
        for (i, l) in self.layers.iter().enumerate() {
            println!(
                " Layer {}: Dense {} -> {} (params {})",
                i,
                l.w.rows,
                l.w.cols,
                l.num_params()
            );
            total += l.num_params();
        }
        println!("Total params: {}", total);
    }
}

/// Helper function to extract a horizontal slice of a tensor's rows.
///
/// Creates a new tensor from rows `start` (inclusive) to `end` (exclusive).
fn slice_rows(t: &Tensor, start: usize, end: usize) -> NeuroxResult<Tensor> {
    assert!(start < end && end <= t.rows);
    let cols = t.cols;
    let mut out = Tensor::zeros(end - start, cols);
    for i in 0..(end - start) {
        for j in 0..cols {
            out.set(i, j, t.get(start + i, j));
        }
    }
    Ok(out)
}
