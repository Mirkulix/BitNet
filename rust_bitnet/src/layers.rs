//! Defines the neural network layers used in the BitNet model architecture.
//!
//! This module includes implementations for:
//! - `BitLinear`: A linear layer that uses 1.58-bit (ternary) quantized weights and int8 quantized activations.
//! - `RMSNorm`: Root Mean Square Layer Normalization.
//! - `Embedding`: Standard token embedding layer.
//!
//! These layers form the building blocks for transformer models.

use ndarray::{Array1, Array2, ArrayView2, Axis};
use crate::quantization::{quantize_activations_int8_per_tensor, quantize_weights_ternary}; // Assuming this will be the full one
use crate::tensor_ops::unpack_i8_to_ternary; // Will be needed for forward pass

// Placeholder for f32 type
type FloatType = f32;
// Placeholder for i8 type
type Int8Type = i8;

/// A linear transformation layer using 1.58-bit (ternary) quantized weights and int8 quantized activations.
///
/// This layer implements the core computation of BitNet. The weights are stored in a packed 2-bit format
/// (4 ternary values per `i8` byte), and activations are quantized to `i8` per input sample (row).
/// The forward pass involves:
/// 1. Quantizing the `f32` input activations to `i8` for each row in the batch.
/// 2. Unpacking the 2-bit ternary weights into an `i8` matrix of `{-1, 0, 1}` values.
/// 3. Performing a matrix multiplication between the quantized activations and unpacked weights,
///    accumulating results in `i32` to prevent overflow.
/// 4. Dequantizing the result by multiplying with the input activation scales and the weight scales.
///
/// Bias is typically not used in BitNet's linear layers.
#[derive(Debug)]
pub struct BitLinear {
    /// Packed 2-bit ternary weights. Dimensions: `(out_features, in_features / 4)`.
    pub weights: Array2<Int8Type>,
    /// Scaling factors for dequantizing the ternary weights. One per output feature.
    pub weight_scales: Array1<FloatType>,
    /// Number of input features. Must be divisible by 4.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
    // pub bias: Option<Array1<FloatType>>, // Usually None for BitNet
}

impl BitLinear {
    /// Creates a new `BitLinear` layer with uninitialized (zeroed) weights and scales.
    ///
    /// Weights and scales should be loaded subsequently using the [`load_weights`](BitLinear::load_weights) method.
    ///
    /// # Arguments
    /// * `in_features`: The number of input features for the layer. Must be divisible by 4
    ///   due to the 2-bit packing of weights.
    /// * `out_features`: The number of output features for the layer.
    ///
    /// # Panics
    /// Panics if `in_features` is not divisible by 4.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        if in_features % 4 != 0 {
            panic!("in_features ({}) must be divisible by 4 for BitLinear due to 2-bit weight packing.", in_features);
        }
        Self {
            weights: Array2::zeros((out_features, in_features / 4)),
            weight_scales: Array1::zeros(out_features),
            in_features,
            out_features,
        }
    }

    /// Loads pre-quantized and packed weights, along with their corresponding scaling factors, into the layer.
    ///
    /// # Arguments
    /// * `quantized_weights`: An `Array2<i8>` containing the packed 2-bit ternary weights.
    ///   Expected dimensions are `(out_features, in_features / 4)`.
    /// * `scales`: An `Array1<f32>` containing the scaling factors for each output feature,
    ///   used to dequantize the weights. Expected length is `out_features`.
    ///
    /// # Panics
    /// Panics if the dimensions of `quantized_weights` or `scales` do not match the layer's configuration.
    pub fn load_weights(&mut self, quantized_weights: Array2<Int8Type>, scales: Array1<FloatType>) {
        assert_eq!(quantized_weights.nrows(), self.out_features, "Mismatch in number of output features for weights.");
        assert_eq!(quantized_weights.ncols(), self.in_features / 4, "Mismatch in number of input features (packed) for weights.");
        assert_eq!(scales.len(), self.out_features, "Mismatch in number of output features for scales.");

        self.weights = quantized_weights;
        self.weight_scales = scales;
    }

    /// Performs the forward pass for the `BitLinear` layer.
    ///
    /// The computation involves quantizing inputs, unpacking weights, performing an integer
    /// matrix multiplication, and then dequantizing the result.
    ///
    /// # Arguments
    /// * `input`: A 2D array view (`ArrayView2<f32>`) representing the batch of input activations.
    ///   Dimensions are `(batch_size, in_features)`.
    ///
    /// # Returns
    /// * `Array2<f32>`: The output of the linear layer, with dimensions `(batch_size, out_features)`.
    ///
    /// # Panics
    /// Panics if the number of columns in `input` does not match `self.in_features`.
    pub fn forward(&self, input: &ArrayView2<FloatType>) -> Array2<FloatType> {
        assert_eq!(input.ncols(), self.in_features, "Input feature dimension mismatch with BitLinear in_features.");
        let batch_size = input.nrows();

        // 1. Quantize activations (f32 -> i8)
        // For now, using per-tensor quantization for activations.
        // The Python code snippet `s = 127 / input.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)`
        // suggests per-row (per-token) quantization for inputs if input is (batch, seq_len, dim)
        // and then reshaped, or per-token if (tokens, dim).
        // If `input` is (batch_size, in_features), per-row quantization might be more appropriate.
        // Let's assume per-tensor for now for simplicity, matching `quantize_activations_int8_per_tensor`.

        // To match the Python `input.abs().max(dim=-1, keepdim=True)` behavior for a (batch, features) input,
        // we'd need to quantize each row of the input independently.

        let mut output = Array2::zeros((batch_size, self.out_features));

        for r in 0..batch_size {
            let row_input = input.row(r);
            let (quantized_input_row, input_scale_row) = quantize_activations_int8_per_tensor(&row_input.insert_axis(Axis(0)).viewD());
            // quantized_input_row is (1, in_features)

            // TODO: This part needs the full implementation of unpack_i8_to_ternary and the actual matmul logic.
            // 2. Unpack weights (packed i8 -> i8 with -1, 0, 1)
            //    This should ideally be done once if weights don't change, or integrated into the matmul.
            //    For now, let's conceptually unpack it.
            //    let unpacked_weights: Array2<Int8Type> = unpack_i8_to_ternary(&self.weights.view(), self.in_features);
            //
            // 3. Perform matrix multiplication with i8 * i8 -> i32 or similar
            //    (quantized_input_row (1, in_features)) @ (unpacked_weights.T (in_features, out_features))
            //    This would result in a (1, out_features) row of i32 accumulators.
            //    let dot_product_result: Array1<i32> = ... ;
            //
            // 4. Dequantize: result_f32 = dot_product_result_i32 * input_scale_row * self.weight_scales
            //    for c in 0..self.out_features {
            //        output[[r,c]] = dot_product_result[c] as FloatType * input_scale_row * self.weight_scales[c];
            //    }

            // Actual forward pass logic:
            // 1. Quantize current input row (already done: quantized_input_row, input_scale_row)
            //    quantized_input_row is (1, in_features) of i8.
            //    We need it as (in_features,) for dot product, or handle shapes in matmul.
            let q_input_flat_view = quantized_input_row.into_shape(self.in_features).unwrap();

            // 2. Unpack weights (done once, or on-the-fly if preferred for memory, but slower)
            //    For simplicity in this step, let's assume we unpack them here.
            //    In a real scenario, this might be part of a more optimized kernel or pre-unpacked.
            let unpacked_weights: Array2<Int8Type> = unpack_i8_to_ternary(&self.weights.view(), self.in_features);
            // unpacked_weights is (out_features, in_features)

            // 3. Perform matrix multiplication: (1, in_features) @ (in_features, out_features) -> (1, out_features)
            //    The result of i8 * i8 can be accumulated in i32 to avoid overflow before scaling.
            let mut dot_product_row = Array1::<i32>::zeros(self.out_features);
            for c_out in 0..self.out_features {
                let mut acc: i32 = 0;
                for c_in in 0..self.in_features {
                    acc += (q_input_flat_view[c_in] as i32) * (unpacked_weights[[c_out, c_in]] as i32);
                }
                dot_product_row[c_out] = acc;
            }

            // 4. Dequantize and store in the output matrix
            //    output_val = accumulator_i32 * input_scale_for_row * weight_scale_for_output_channel
            let mut output_row = output.row_mut(r);
            for c_out in 0..self.out_features {
                output_row[c_out] = (dot_product_row[c_out] as FloatType) * input_scale_row * self.weight_scales[c_out];
            }
        }

        // println!("BitLinear forward pass (partially implemented, matmul and dequant logic pending)");
        output
    }
}

/// Root Mean Square Layer Normalization.
///
/// RMSNorm normalizes the activations of a layer by their root mean square,
/// and then scales them by a learnable weight parameter. It is a simpler
/// alternative to LayerNorm, as it does not involve learnable bias or mean centering.
///
/// Formula: `output = (input / sqrt(mean(input^2) + eps)) * weight`
/// Normalization is typically applied over the last dimension (features/embedding dimension).
#[derive(Debug)]
pub struct RMSNorm {
    /// The dimension of the features to normalize.
    dim: usize,
    /// The learnable scaling parameters (gamma).
    weight: Array1<FloatType>,
    /// A small constant added to the denominator for numerical stability.
    eps: FloatType,
}

impl RMSNorm {
    /// Creates a new `RMSNorm` layer.
    ///
    /// The learnable weight parameter is initialized to ones. `eps` is a small
    /// constant added to the denominator for numerical stability.
    ///
    /// # Arguments
    /// * `dim`: The dimension of the features to normalize (typically the last dimension).
    /// * `eps`: A small float constant for numerical stability.
    pub fn new(dim: usize, eps: FloatType) -> Self {
        Self {
            dim,
            weight: Array1::ones(dim), // Initialize weights to 1.0
            eps,
        }
    }

    /// Loads pre-trained weights into the RMSNorm layer.
    ///
    /// # Arguments
    /// * `weights`: An `Array1<f32>` containing the learnable scaling parameters (gamma).
    ///   The length of this array must match the `dim` of the layer.
    ///
    /// # Panics
    /// Panics if `weights.len()` does not match `self.dim`.
    pub fn load_weights(&mut self, weights: Array1<FloatType>) {
        assert_eq!(weights.len(), self.dim, "RMSNorm weight dimension mismatch.");
        self.weight = weights;
    }

    /// Performs the forward pass for RMSNorm.
    ///
    /// Normalizes the input over its last dimension.
    ///
    /// # Arguments
    /// * `input`: A 2D array view (`ArrayView2<f32>`) with dimensions `(batch_size, dim)`.
    ///   While RMSNorm can be applied to tensors of higher rank, this implementation
    ///   currently expects a 2D input for simplicity.
    ///
    /// # Returns
    /// * `Array2<f32>`: The normalized and scaled output, with the same dimensions as the input.
    ///
    /// # Panics
    /// Panics if `input.ncols()` does not match `self.dim`.
    pub fn forward(&self, input: &ArrayView2<FloatType>) -> Array2<FloatType> {
        assert_eq!(input.ncols(), self.dim, "Input feature dimension mismatch for RMSNorm.");

        let mut output = Array2::zeros(input.raw_dim());

        for (i, row) in input.outer_iter().enumerate() {
            let sum_sq = row.iter().map(|&x| x * x).sum::<FloatType>();
            let rms = (sum_sq / self.dim as FloatType + self.eps).sqrt();

            let mut out_row = output.row_mut(i);
            for (j, val) in row.iter().enumerate() {
                out_row[j] = (val / rms) * self.weight[j];
            }
        }
        output
    }
}

/// Standard token embedding layer.
///
/// This layer maps discrete token IDs to dense vector representations (embeddings).
/// It stores a weight matrix of shape `(num_embeddings, embedding_dim)`, where each row
/// corresponds to the embedding vector for a token ID.
#[derive(Debug)]
pub struct Embedding {
    /// The size of the vocabulary (number of possible token IDs).
    num_embeddings: usize,
    /// The dimensionality of the embedding vectors.
    embedding_dim: usize,
    /// The learnable weight matrix holding the embedding vectors.
    weight: Array2<FloatType>,
}

impl Embedding {
    /// Creates a new `Embedding` layer.
    ///
    /// The embedding weights are initialized with small random values drawn from a
    /// uniform distribution `U(-0.01, 0.01)`. In a typical workflow, these weights
    /// would be overwritten by loading a pre-trained checkpoint.
    ///
    /// # Arguments
    /// * `num_embeddings`: The size of the vocabulary.
    /// * `embedding_dim`: The desired dimensionality of the embedding vectors.
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        // Requires ndarray-rand, ensure it's in [dev-dependencies] or [dependencies] if used outside tests
        let weight = Array::random((num_embeddings, embedding_dim), Uniform::new(-0.01, 0.01));
        Self {
            num_embeddings,
            embedding_dim,
            weight,
        }
    }

    /// Loads pre-trained embedding weights into the layer.
    ///
    /// # Arguments
    /// * `weights`: An `Array2<f32>` containing the embedding vectors.
    ///   Expected dimensions are `(num_embeddings, embedding_dim)`.
    ///
    /// # Panics
    /// Panics if the dimensions of `weights` do not match the layer's configuration.
    pub fn load_weights(&mut self, weights: Array2<FloatType>) {
        assert_eq!(weights.nrows(), self.num_embeddings, "Embedding weight num_embeddings mismatch.");
        assert_eq!(weights.ncols(), self.embedding_dim, "Embedding weight embedding_dim mismatch.");
        self.weight = weights;
    }

    /// Performs the embedding lookup for a batch of token IDs.
    ///
    /// # Arguments
    /// * `input_ids`: A 1D array view (`ArrayView1<usize>`) of token IDs.
    ///
    /// # Returns
    /// * `Array2<f32>`: An array containing the corresponding embedding vectors.
    ///   The dimensions are `(input_ids.len(), embedding_dim)`.
    ///
    /// # Panics
    /// Panics if any `token_id` in `input_ids` is out of bounds (i.e., `>= num_embeddings`).
    pub fn forward(&self, input_ids: &ArrayView1<usize>) -> Array2<FloatType> {
        let batch_size = input_ids.len();
        let mut output = Array2::zeros((batch_size, self.embedding_dim));

        for (i, &token_id) in input_ids.iter().enumerate() {
            if token_id >= self.num_embeddings {
                panic!("Token ID {} out of bounds for num_embeddings {}", token_id, self.num_embeddings);
            }
            output.row_mut(i).assign(&self.weight.row(token_id));
        }
        output
    }
}

// Other layers like Embedding, Attention, FeedForward will also be defined here.

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, Array, Ix2};
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    #[test]
    fn test_bitlinear_new() {
        let bl = BitLinear::new(16, 32); // 16 in_features, 32 out_features
        assert_eq!(bl.in_features, 16);
        assert_eq!(bl.out_features, 32);
        assert_eq!(bl.weights.shape(), &[32, 16 / 4]);
        assert_eq!(bl.weight_scales.shape(), &[32]);
    }

    #[test]
    fn test_bitlinear_forward_logic() {
        let in_features = 4;
        let out_features = 2;
        let mut bitlinear = BitLinear::new(in_features, out_features);

        // --- Setup Weights ---
        // Weight matrix (f32):
        // [[w11, w12, w13, w14],  ; out_feature 0
        //  [w21, w22, w23, w24]]  ; out_feature 1
        // Let's use values that are easy to quantize and track
        // Row 0: [1.0, 0.1, -0.8, 0.9] -> scale0 = 0.7, ternary0 = [1, 0, -1, 1], packed0 = 97
        // Row 1: [0.2, -0.8, -0.7, 0.1] -> scale1 = 0.45, ternary1 = [0, -1, -1, 0], packed1 = 40
        let weights_f32 = arr2(&[
            [1.0, 0.1, -0.8, 0.9],
            [0.2, -0.8, -0.7, 0.1]
        ]);
        let (packed_w, scales_w) = crate::quantization::quantize_weights_ternary(&weights_f32.view());
        // packed_w should be [[97], [40]]
        // scales_w should be [0.7, 0.45]
        bitlinear.load_weights(packed_w, scales_w);

        // --- Setup Input ---
        // Input (batch_size=1, in_features=4)
        // Let input be [10.0, 20.0, -30.0, 0.0]
        // abs_max_input = 30.0. scale_input = 30.0 / 127.0 approx 0.23622
        // q_input = round(input / scale_input)
        // q_input = [round(10.0/0.23622), round(20.0/0.23622), round(-30.0/0.23622), round(0.0/0.23622)]
        //         = [round(42.33), round(84.66), round(-127.0), round(0.0)]
        //         = [42, 85, -127, 0] (as i8)
        let input_f32 = arr2(&[[10.0, 20.0, -30.0, 0.0]]);
        let (expected_q_input, expected_input_scale) =
            crate::quantization::quantize_activations_int8_per_tensor(&input_f32.viewD());
        // expected_q_input (dyn) = [[42, 85, -127, 0]]
        // expected_input_scale = 30.0 / 127.0

        // --- Expected dot products (i32) ---
        // Unpacked weights:
        // ternary_w = [[ 1,  0, -1,  1],
        //              [ 0, -1, -1,  0]]
        // q_input_flat = [42, 85, -127, 0]

        // Dot for output channel 0:
        // (42*1) + (85*0) + (-127*-1) + (0*1) = 42 + 0 + 127 + 0 = 169
        // Dot for output channel 1:
        // (42*0) + (85*-1) + (-127*-1) + (0*0) = 0 - 85 + 127 + 0 = 42
        let expected_dot_products = [169, 42];

        // --- Expected final output (f32) ---
        // output[0] = dot_product[0] * input_scale * weight_scale[0]
        //           = 169 * (30.0/127.0) * 0.7
        //           = 169 * 0.23622047 * 0.7 = 169 * 0.16535433 = 27.94488177
        // output[1] = dot_product[1] * input_scale * weight_scale[1]
        //           = 42 * (30.0/127.0) * 0.45
        //           = 42 * 0.23622047 * 0.45 = 42 * 0.10629921 = 4.46456682
        let expected_output_f32 = arr2(&[[
            169.0 * (30.0/127.0) * 0.7,
            42.0 * (30.0/127.0) * 0.45
        ]]);

        // --- Perform forward pass ---
        let output = bitlinear.forward(&input_f32.view());

        assert_eq!(output.shape(), &[1, out_features]);
        // Compare with a tolerance due to floating point arithmetic
        for r in 0..output.nrows() {
            for c in 0..output.ncols() {
                assert!((output[[r,c]] - expected_output_f32[[r,c]]).abs() < 1e-5,
                        "Output at [{},{}] mismatch: {} vs {}", r, c, output[[r,c]], expected_output_f32[[r,c]]);
            }
        }
    }


    #[test]
    #[should_panic]
    fn test_bitlinear_new_panic_in_features_not_divisible_by_4() {
        BitLinear::new(15, 32);
    }

    #[test]
    fn test_bitlinear_load_weights() {
        let mut bl = BitLinear::new(8, 4);
        let weights_data = Array2::from_elem((4, 8 / 4), 1i8);
        let scales_data = Array1::from_elem(4, 0.5f32);
        bl.load_weights(weights_data.clone(), scales_data.clone());

        assert_eq!(bl.weights, weights_data);
        assert_eq!(bl.weight_scales, scales_data);
    }

    #[test]
    #[should_panic]
    fn test_bitlinear_load_weights_panic_nrows() {
        let mut bl = BitLinear::new(8, 4);
        let weights_data = Array2::from_elem((3, 2), 1i8); // Wrong nrows
        let scales_data = Array1::from_elem(4, 0.5f32);
        bl.load_weights(weights_data, scales_data);
    }

    #[test]
    #[should_panic]
    fn test_bitlinear_load_weights_panic_ncols() {
        let mut bl = BitLinear::new(8, 4);
        let weights_data = Array2::from_elem((4, 1), 1i8); // Wrong ncols (should be 8/4=2)
        let scales_data = Array1::from_elem(4, 0.5f32);
        bl.load_weights(weights_data, scales_data);
    }

    #[test]
    #[should_panic]
    fn test_bitlinear_load_weights_panic_scales_len() {
        let mut bl = BitLinear::new(8, 4);
        let weights_data = Array2::from_elem((4, 2), 1i8);
        let scales_data = Array1::from_elem(3, 0.5f32); // Wrong len
        bl.load_weights(weights_data, scales_data);
    }

    #[test]
    fn test_bitlinear_forward_placeholder() {
        // This test mainly checks that the forward pass can be called with correct dimensions
        // and returns a correctly shaped zero array, as the actual logic is pending.
        let mut bl = BitLinear::new(8, 4);
        // Load dummy weights and scales, otherwise calculations might involve uninitialized values if we went further.
        let dummy_q_weights = Array2::zeros((4, 8/4));
        let dummy_scales = Array1::zeros(4);
        bl.load_weights(dummy_q_weights, dummy_scales);

        let input_data = arr2(&[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                                [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]]); // Batch size 2
        let output = bl.forward(&input_data.view());

        assert_eq!(output.shape(), &[2, 4]); // Batch size 2, out_features 4
        // As the forward pass is a placeholder returning zeros:
        assert!(output.iter().all(|&x| x == 0.0));
    }

    #[test]
    #[should_panic]
    fn test_bitlinear_forward_panic_input_features_mismatch() {
        let bl = BitLinear::new(8, 4);
        let input_data = Array2::<FloatType>::zeros((2, 7)); // Incorrect in_features
        bl.forward(&input_data.view());
    }

    #[test]
    fn test_rmsnorm_new() {
        let norm = RMSNorm::new(16, 1e-5);
        assert_eq!(norm.dim, 16);
        assert_eq!(norm.eps, 1e-5);
        assert_eq!(norm.weight.len(), 16);
        assert!(norm.weight.iter().all(|&w| (w - 1.0).abs() < FloatType::EPSILON));
    }

    #[test]
    fn test_rmsnorm_load_weights() {
        let mut norm = RMSNorm::new(4, 1e-5);
        let new_weights = arr1(&[0.1, 0.2, 0.3, 0.4]);
        norm.load_weights(new_weights.clone());
        assert_eq!(norm.weight, new_weights);
    }

    #[test]
    #[should_panic]
    fn test_rmsnorm_load_weights_panic_dim() {
        let mut norm = RMSNorm::new(4, 1e-5);
        let new_weights = arr1(&[0.1, 0.2, 0.3]); // Wrong dim
        norm.load_weights(new_weights);
    }

    #[test]
    fn test_rmsnorm_forward() {
        let dim = 3;
        let eps = 1e-6; // Use a slightly larger eps for float comparisons
        let mut norm = RMSNorm::new(dim, eps);
        norm.load_weights(arr1(&[1.0, 1.0, 1.0])); // Use identity weights for easier checking

        let input = arr2(&[[1.0, 2.0, 3.0],  // RMS = sqrt((1+4+9)/3) = sqrt(14/3) = sqrt(4.666) = 2.1602
                           [4.0, 5.0, 6.0]]); // RMS = sqrt((16+25+36)/3) = sqrt(77/3) = sqrt(25.666) = 5.0662

        let output = norm.forward(&input.view());

        // Row 1:
        // rms1 = ((1.0^2 + 2.0^2 + 3.0^2) / 3.0 + eps).sqrt() = (14.0/3.0 + eps).sqrt()
        let rms1 = ((1.0*1.0 + 2.0*2.0 + 3.0*3.0) / dim as FloatType + eps).sqrt();
        let expected_row1 = arr1(&[1.0/rms1, 2.0/rms1, 3.0/rms1]);

        // Row 2:
        // rms2 = ((4.0^2 + 5.0^2 + 6.0^2) / 3.0 + eps).sqrt() = (77.0/3.0 + eps).sqrt()
        let rms2 = ((4.0*4.0 + 5.0*5.0 + 6.0*6.0) / dim as FloatType + eps).sqrt();
        let expected_row2 = arr1(&[4.0/rms2, 5.0/rms2, 6.0/rms2]);

        for i in 0..dim {
            assert!((output[[0, i]] - expected_row1[i]).abs() < 1e-5);
            assert!((output[[1, i]] - expected_row2[i]).abs() < 1e-5);
        }

        // Test with different weights
        norm.load_weights(arr1(&[0.5, 1.0, 1.5]));
        let output_scaled = norm.forward(&input.view());
        for i in 0..dim {
            assert!((output_scaled[[0, i]] - expected_row1[i] * norm.weight[i]).abs() < 1e-5);
            assert!((output_scaled[[1, i]] - expected_row2[i] * norm.weight[i]).abs() < 1e-5);
        }
    }

    #[test]
    #[should_panic]
    fn test_rmsnorm_forward_panic_input_dim_mismatch() {
        let norm = RMSNorm::new(3, 1e-5);
        let input_data = Array2::<FloatType>::zeros((2, 2)); // Incorrect dim
        norm.forward(&input_data.view());
    }

    #[test]
    fn test_rmsnorm_forward_zeros_input() {
        let dim = 3;
        let eps = 1e-6;
        let norm = RMSNorm::new(dim, eps); // weights are 1.0
        let input = Array2::<FloatType>::zeros((2, dim));
        let output = norm.forward(&input.view());

        // When input is all zeros, sum_sq is 0. rms = sqrt(eps).
        // Output should be (0 / sqrt(eps)) * 1.0 = 0.
        assert!(output.iter().all(|&x| x.abs() < 1e-9)); // Should be very close to 0
    }

    #[test]
    fn test_embedding_new() {
        let num_embeddings = 100;
        let embedding_dim = 10;
        let embed = Embedding::new(num_embeddings, embedding_dim);
        assert_eq!(embed.num_embeddings, num_embeddings);
        assert_eq!(embed.embedding_dim, embedding_dim);
        assert_eq!(embed.weight.shape(), &[num_embeddings, embedding_dim]);
    }

    #[test]
    fn test_embedding_load_weights() {
        let num_embeddings = 10;
        let embedding_dim = 4;
        let mut embed = Embedding::new(num_embeddings, embedding_dim);
        let new_weights = Array::from_shape_fn((num_embeddings, embedding_dim), |(i, j)| (i * embedding_dim + j) as FloatType);
        embed.load_weights(new_weights.clone());
        assert_eq!(embed.weight, new_weights);
    }

    #[test]
    #[should_panic]
    fn test_embedding_load_weights_panic_num_embeddings() {
        let mut embed = Embedding::new(10, 4);
        let new_weights = Array2::<FloatType>::zeros((9, 4)); // Wrong num_embeddings
        embed.load_weights(new_weights);
    }

    #[test]
    #[should_panic]
    fn test_embedding_load_weights_panic_embedding_dim() {
        let mut embed = Embedding::new(10, 4);
        let new_weights = Array2::<FloatType>::zeros((10, 3)); // Wrong embedding_dim
        embed.load_weights(new_weights);
    }

    #[test]
    fn test_embedding_forward() {
        let num_embeddings = 5;
        let embedding_dim = 3;
        let mut embed = Embedding::new(num_embeddings, embedding_dim);
        // Create easily identifiable weights
        let weights_data = Array::from_shape_fn((num_embeddings, embedding_dim), |(i, j)| (i * 10 + j) as FloatType);
        // e.g., [[0,1,2], [10,11,12], [20,21,22], ...]
        embed.load_weights(weights_data);

        let input_ids = arr1(&[0, 2, 1]);
        let output = embed.forward(&input_ids.view());

        let expected_output = arr2(&[[0.0, 1.0, 2.0],         // Embedding of ID 0
                                     [20.0, 21.0, 22.0],      // Embedding of ID 2
                                     [10.0, 11.0, 12.0]]);    // Embedding of ID 1

        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_embedding_forward_batch() {
        let num_embeddings = 10;
        let embedding_dim = 2;
        let mut embed = Embedding::new(num_embeddings, embedding_dim);
        let weights_data = Array::from_shape_fn((num_embeddings, embedding_dim), |(i,j)| (i as FloatType + j as FloatType * 0.1));
        embed.load_weights(weights_data);

        let input_ids = arr1(&[0, 1, 2, 0, 3]);
        let output = embed.forward(&input_ids.view());

        assert_eq!(output.nrows(), input_ids.len());
        assert_eq!(output.ncols(), embedding_dim);

        for (i, &token_id) in input_ids.iter().enumerate() {
            let expected_row = embed.weight.row(token_id);
            assert_eq!(output.row(i), expected_row);
        }
    }


    #[test]
    #[should_panic]
    fn test_embedding_forward_panic_id_out_of_bounds() {
        let embed = Embedding::new(5, 3);
        let input_ids = arr1(&[0, 5]); // 5 is out of bounds
        embed.forward(&input_ids.view());
    }
}
