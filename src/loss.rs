use crate::tensor::Tensor;

/// Mean Squared Error loss and gradient. inputs are (batch x features)
pub fn mse_loss(pred: &Tensor, target: &Tensor) -> (f32, Tensor) {
    assert_eq!(pred.rows, target.rows);
    assert_eq!(pred.cols, target.cols);
    let mut sum = 0.0;
    let mut grad = vec![0.0; pred.data.len()];
    for (i, g) in grad.iter_mut().enumerate().take(pred.data.len()) {
        let diff = pred.data[i] - target.data[i];
        sum += diff * diff;
        *g = 2.0 * diff / (pred.rows as f32); // averaged over batch
    }
    (
        sum / (pred.rows as f32),
        Tensor::from_data(grad, pred.rows, pred.cols),
    )
}

/// Cross-entropy (assumes softmax already applied). target is one-hot or probabilities.
/// returns (loss, grad wrt logits after softmax)
pub fn cross_entropy_loss(prob: &Tensor, target: &Tensor) -> (f32, Tensor) {
    assert_eq!(prob.rows, target.rows);
    assert_eq!(prob.cols, target.cols);
    let mut loss = 0.0;
    let mut grad = vec![0.0; prob.data.len()];
    for i in 0..prob.rows {
        for j in 0..prob.cols {
            let p = (prob.get(i, j)).max(1e-7);
            let t = target.get(i, j);
            loss -= t * p.ln();
            grad[i * prob.cols + j] = (p - t) / (prob.rows as f32); // average over batch
        }
    }
    (loss, Tensor::from_data(grad, prob.rows, prob.cols))
}
