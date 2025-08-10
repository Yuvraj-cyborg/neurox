use neurox::{Model, Tensor};

#[test]
fn test_tensor_creation() {
    let tensor = Tensor::new(2, 3);
    assert_eq!(tensor.rows, 2);
    assert_eq!(tensor.cols, 3);
    assert_eq!(tensor.data.len(), 6);
    assert!(tensor.data.iter().all(|&x| x == 0.0));
}

#[test]
fn test_tensor_get_set() {
    let mut tensor = Tensor::new(2, 2);
    tensor.set(0, 1, 5.5);
    assert_eq!(tensor.get(0, 1), 5.5);
}

#[test]
fn test_matmul_shapes() {
    let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
    let c = neurox::ops::matmul(&a, &b);
    assert_eq!(c.rows, 2);
    assert_eq!(c.cols, 2);
    assert_eq!(c.get(0, 0), 19.0);
    assert_eq!(c.get(0, 1), 22.0);
    assert_eq!(c.get(1, 0), 43.0);
    assert_eq!(c.get(1, 1), 50.0);
}

#[test]
fn test_relu_activation() {
    let t = Tensor::from_data(vec![-1.0, 0.0, 2.5, -3.2], 2, 2);
    let r = neurox::ops::relu(&t);
    assert_eq!(r.get(0, 0), 0.0);
    assert_eq!(r.get(0, 1), 0.0);
    assert_eq!(r.get(1, 0), 2.5);
    assert_eq!(r.get(1, 1), 0.0);
}

#[test]
fn test_model_forward_output_shape() {
    let model = Model::new(&[3, 4, 2]);
    let input = Tensor::from_data(vec![1.0, 2.0, 3.0], 1, 3);
    let output = model.forward(input);
    assert_eq!(output.rows, 1);
    assert_eq!(output.cols, 2);
}
