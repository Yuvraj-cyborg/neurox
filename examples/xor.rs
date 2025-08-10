use neurox::activations;
use neurox::layers::Activation;
use neurox::loss;
use neurox::{Model, Tensor};

fn main() {
    // model: 2 -> 6 -> 2
    let mut model = Model::new(&[2, 6, 2], Activation::ReLU);

    // XOR inputs (4 samples x 2 features)
    let inputs = Tensor::from_data(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], 4, 2);

    // one-hot targets: class 0 -> [1,0], class 1 -> [0,1]
    let targets = Tensor::from_data(vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], 4, 2);

    println!("Starting XOR training (small network)...");
    model
        .train_sgd(&inputs, &targets, 600, 4, 0.1)
        .expect("training failed");

    // Evaluate
    let preds = model.forward(&inputs).expect("forward failed");
    let probs = activations::softmax(&preds);

    // Print per-sample predicted class and probability

    for i in 0..probs.rows {
        let mut best = 0usize;
        let mut best_p = -1.0f32;
        for j in 0..probs.cols {
            let p = probs.get(i, j);
            if p > best_p {
                best = j;
                best_p = p;
            }
        }
        println!("Sample {} -> class {} (p={:.4})", i, best, best_p);
    }

    let (final_loss, _) = loss::cross_entropy_loss(&probs, &targets);
    println!("Final cross-entropy loss: {:.6}", final_loss);
}
