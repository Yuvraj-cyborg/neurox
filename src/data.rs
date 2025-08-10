//! Provides utilities for data loading and manipulation.

use crate::errors::{NeuroxError, NeuroxResult};
use crate::tensor::Tensor;
use std::fs::File;
use std::io::{BufRead, BufReader};

/// Loads a `Tensor` from a CSV file.
///
/// Assumes a rectangular grid of floating-point numbers with no header.
/// Empty lines are skipped. Parsing errors default the value to `0.0`.
///
/// # Errors
///
/// Returns `NeuroxError::Io` on file-related issues or `NeuroxError::InvalidArgument`
/// if the CSV is empty or malformed.
pub fn tensor_from_csv(path: &str) -> NeuroxResult<Tensor> {
    let f = File::open(path)?;
    let reader = BufReader::new(f);
    let mut rows: Vec<Vec<f32>> = Vec::new();
    for line in reader.lines() {
        let l = line?;
        if l.trim().is_empty() {
            continue;
        }
        let vals: Vec<f32> = l
            .split(',')
            .map(|s| s.trim().parse::<f32>().unwrap_or(0.0))
            .collect();
        rows.push(vals);
    }
    let r = rows.len();
    if r == 0 {
        return Err(NeuroxError::InvalidArgument("empty csv".into()));
    }
    let c = rows[0].len();
    let mut data = Vec::with_capacity(r * c);
    for row in rows {
        if row.len() != c {
            return Err(NeuroxError::InvalidArgument(
                "csv has non-uniform row length".into(),
            ));
        }
        data.extend_from_slice(&row);
    }
    Ok(Tensor::from_data(data, r, c))
}

/// Splits a tensor's rows into two tensors for training and testing.
///
/// This performs a simple sequential split. The first `ratio` proportion of rows
/// go into the training set, and the remainder go into the test set.
///
/// # Panics
///
/// Panics if `ratio` is not between `0.0` and `1.0`.
pub fn train_test_split(t: &Tensor, ratio: f32) -> NeuroxResult<(Tensor, Tensor)> {
    assert!(
        ratio > 0.0 && ratio < 1.0,
        "Split ratio must be between 0.0 and 1.0"
    );
    let n = t.rows;
    let train_n = ((n as f32) * ratio).round() as usize;
    let train = slice_rows(t, 0, train_n)?;
    let test = slice_rows(t, train_n, n)?;
    Ok((train, test))
}

/// Helper to extract a horizontal slice of a tensor.
fn slice_rows(t: &Tensor, start: usize, end: usize) -> NeuroxResult<Tensor> {
    assert!(start <= end && end <= t.rows);
    let cols = t.cols;
    let mut out = Tensor::zeros(end - start, cols);
    for i in 0..(end - start) {
        for j in 0..cols {
            out.set(i, j, t.get(start + i, j));
        }
    }
    Ok(out)
}
