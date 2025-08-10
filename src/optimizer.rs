//! Provides optimization algorithms for updating model parameters.

use crate::layers::Dense;

/// A simple Stochastic Gradient Descent (SGD) optimizer.
///
/// This implementation does not include momentum for simplicity.
pub struct SGD {
    pub lr: f32,
}

impl SGD {
    /// Creates a new SGD optimizer with a given learning rate.
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }

    /// Performs a single optimization step, updating the parameters of all layers.
    pub fn step(&self, layers: &mut [Dense]) {
        for l in layers {
            l.apply_gradients(self.lr);
        }
    }
}

/// The Adam optimization algorithm.
///
/// Adam maintains per-parameter adaptive learning rates from estimates of
/// first and second moments of the gradients.
pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub t: usize,
    // Per-layer moving averages for weights
    pub m_w: Vec<Vec<f32>>,
    pub v_w: Vec<Vec<f32>>,
    // Per-layer moving averages for biases
    pub m_b: Vec<Vec<f32>>,
    pub v_b: Vec<Vec<f32>>,
}

impl Adam {
    /// Creates a new Adam optimizer and initializes its state vectors.
    ///
    /// # Arguments
    /// * `lr` - The learning rate.
    /// * `layers` - A reference to the model's layers, used to initialize state vectors
    ///   with the correct dimensions.
    pub fn new(lr: f32, layers: &Vec<Dense>) -> Self {
        let mut m_w = Vec::new();
        let mut v_w = Vec::new();
        let mut m_b = Vec::new();
        let mut v_b = Vec::new();
        for l in layers {
            m_w.push(vec![0.0; l.w.data.len()]);
            v_w.push(vec![0.0; l.w.data.len()]);
            m_b.push(vec![0.0; l.b.data.len()]);
            v_b.push(vec![0.0; l.b.data.len()]);
        }
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            m_w,
            v_w,
            m_b,
            v_b,
        }
    }

    /// Performs a single Adam optimization step.
    pub fn step(&mut self, layers: &mut [Dense]) {
        self.t += 1;
        for (li, l) in layers.iter_mut().enumerate() {
            if l.grad_w.is_none() || l.grad_b.is_none() {
                continue;
            }
            let gw = l.grad_w.as_ref().unwrap();
            let gb = l.grad_b.as_ref().unwrap();

            // Update weights
            for i in 0..l.w.data.len() {
                let g = gw.data[i];
                // Update biased first moment estimate
                let m = &mut self.m_w[li][i];
                *m = self.beta1 * (*m) + (1.0 - self.beta1) * g;
                // Update biased second raw moment estimate
                let v = &mut self.v_w[li][i];
                *v = self.beta2 * (*v) + (1.0 - self.beta2) * (g * g);
                // Compute bias-corrected first and second moment estimates
                let m_hat = (*m) / (1.0 - self.beta1.powi(self.t as i32));
                let v_hat = (*v) / (1.0 - self.beta2.powi(self.t as i32));
                // Update parameter
                l.w.data[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }

            // Update biases
            for i in 0..l.b.data.len() {
                let g = gb.data[i];
                let m = &mut self.m_b[li][i];
                let v = &mut self.v_b[li][i];
                *m = self.beta1 * (*m) + (1.0 - self.beta1) * g;
                *v = self.beta2 * (*v) + (1.0 - self.beta2) * (g * g);
                let m_hat = (*m) / (1.0 - self.beta1.powi(self.t as i32));
                let v_hat = (*v) / (1.0 - self.beta2.powi(self.t as i32));
                l.b.data[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
        }
    }
}
