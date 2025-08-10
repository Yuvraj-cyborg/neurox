use neurox::activations;
use neurox::layers::Activation;
use neurox::loss;
use neurox::ops;
use neurox::{Model, Tensor};

#[test]
fn tensor_create_and_access() {
    let t = Tensor::new(2, 3);
    assert_eq!(t.shape(), (2, 3));
    // all zeros initially
    assert!(t.data.iter().all(|&v| v == 0.0));
}

#[test]
fn matmul_basic() {
    let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
    let c = ops::matmul(&a, &b).expect("matmul failed");
    assert_eq!(c.shape(), (2, 2));
    assert_eq!(c.get(0, 0), 19.0);
    assert_eq!(c.get(0, 1), 22.0);
    assert_eq!(c.get(1, 0), 43.0);
    assert_eq!(c.get(1, 1), 50.0);
}

#[test]
fn relu_activation() {
    let t = Tensor::from_data(vec![-1.0, 0.0, 2.5, -3.2], 2, 2);
    let r = activations::relu(&t);
    assert_eq!(r.get(0, 0), 0.0);
    assert_eq!(r.get(0, 1), 0.0);
    assert_eq!(r.get(1, 0), 2.5);
    assert_eq!(r.get(1, 1), 0.0);
}

#[test]
fn model_forward_shape() {
    let mut model = Model::new(&[3, 4, 2], Activation::ReLU);
    let input = Tensor::from_data(vec![1.0, 2.0, 3.0], 1, 3);
    let out = model.forward(&input).expect("forward failed");
    assert_eq!(out.shape(), (1, 2));
}

#[test]
fn training_reduces_loss() {
    // small XOR-like dataset with one-hot targets (4 samples)
    let inputs = Tensor::from_data(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], 4, 2);

    let targets = Tensor::from_data(vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], 4, 2);

    let mut model = Model::new(&[2, 6, 2], Activation::ReLU);

    // loss before
    let preds_before = model.forward(&inputs).expect("forward before failed");
    let probs_before = activations::softmax(&preds_before);
    let (loss_before, _) = loss::cross_entropy_loss(&probs_before, &targets);

    // train a few epochs
    model
        .train_sgd(&inputs, &targets, 200, 4, 0.1)
        .expect("training failed");

    // after
    let preds_after = model.forward(&inputs).expect("forward after failed");
    let probs_after = activations::softmax(&preds_after);
    let (loss_after, _) = loss::cross_entropy_loss(&probs_after, &targets);

    assert!(
        loss_after < loss_before,
        "loss did not decrease (before: {}, after: {})",
        loss_before,
        loss_after
    );
}
