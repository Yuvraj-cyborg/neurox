use rand::{rng, Rng};

/// A 2D tensor representing a matrix of `f32` values.
///
/// Tensors are stored in a contiguous, row-major layout.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Tensor {
    /// Creates a new tensor of `rows` x `cols` initialized with zeros.
    ///
    /// # Arguments
    ///
    /// * `rows` - The number of rows.
    /// * `cols` - The number of columns.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Creates a new tensor from an existing data vector.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` is not equal to `rows * cols`.
    pub fn from_data(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols, "Data size must match tensor dimensions.");
        Self { data, rows, cols }
    }

    /// Creates a new tensor with random values sampled from a uniform distribution between -1.0 and 1.0.
    pub fn random(rows: usize, cols: usize) -> Self {
        let mut rng = rng();
        let data: Vec<f32> = (0..rows * cols).map(|_| rng.random_range(-1.0..1.0)).collect();
        Self { data, rows, cols }
    }

    /// Returns the value at the specified `(row, col)` index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }

    /// Sets the `value` at the specified `(row, col)` index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn set(&mut self, r: usize, c: usize, value: f32) {
        self.data[r * self.cols + c] = value;
    }
}