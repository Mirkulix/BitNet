//! Handles the quantization of weights and activations.
//!
//! This module provides functions for:
//! - Ternary weight quantization (1.58-bit scheme representation).
//! - Int8 activation quantization.
//! These are crucial for implementing memory and computationally efficient layers like `BitLinear`.

use ndarray::{ArrayView2, Array2, Array1, ArrayViewD, ArrayD, Axis, Zip};
use num_traits::{AsPrimitive, Float};

// Placeholder for f32 type, can be made generic later if needed
type FloatType = f32;
// Placeholder for i8 type
type Int8Type = i8;


/// Quantizes `f32` weights to ternary `{-1, 0, 1}` and then packs them into 2-bit representations within an `i8` array.
///
/// This function implements a quantization scheme inspired by the BitNet paper,
/// specifically for representing weights using approximately 1.58 bits.
/// The process involves:
/// 1. Calculating a per-row (output channel) scaling factor as `mean(abs(row_weights))`.
/// 2. Scaling each weight in a row by its corresponding scaling factor.
/// 3. Ternarizing the scaled weights: values `> 0.5` map to `1`, values `< -0.5` map to `-1`, and others map to `0`.
/// 4. Packing four such ternary values into a single `i8` byte. Each ternary value uses 2 bits:
///    - `0` is represented as `00` (binary).
///    - `1` is represented as `01` (binary).
///    - `-1` is represented as `10` (binary).
///    The first ternary value in a group of four occupies the least significant 2 bits of the byte.
///
/// # Arguments
/// * `weights`: A 2D array view (`ArrayView2<f32>`) of the original `f32` weights.
///              The number of columns must be divisible by 4 to allow packing.
///
/// # Returns
/// A tuple containing:
///   - `Array2<i8>`: The packed 2-bit quantized weights. Each `i8` element holds 4 ternary weights.
///                   The dimensions are `(weights.nrows(), weights.ncols() / 4)`.
///   - `Array1<f32>`: The scaling factors calculated for each row (output feature) of the original weight matrix.
///                   These scales are `mean(abs(row_weights))` and are used for dequantization.
///
/// # Panics
/// Panics if the number of columns in the input `weights` array is not divisible by 4.
pub fn quantize_weights_ternary(
    weights: &ArrayView2<FloatType>
) -> (Array2<Int8Type>, Array1<FloatType>) {
    // TODO: Implement actual ternary quantization and 2-bit packing.
    // This will be more involved. For now, placeholder.
    let num_rows = weights.nrows();
    let num_cols = weights.ncols();

    if num_cols == 0 {
        return (Array2::zeros((num_rows, 0)), Array1::zeros(num_rows));
    }
    if num_cols % 4 != 0 {
        // This is a simplification for now. Real implementation might need padding.
        panic!("Number of columns must be divisible by 4 for 2-bit packing.");
    }

    let packed_weights = Array2::zeros((num_rows, num_cols / 4));
    let scales = Array1::zeros(num_rows); // Placeholder for scales

    // Example of how one might calculate scales (e.g., absmax per row)
    // for (r, row) in weights.outer_iter().enumerate() {
    //     let abs_max = row.iter().map(|&x| x.abs()).fold(FloatType::NEG_INFINITY, FloatType::max);
    //     scales[r] = if abs_max == 0.0 { 1.0 } else { abs_max / 1.0 }; // Assuming ternary range is -1, 0, 1
    // }

    // Actual ternary quantization and packing logic:
    // 1. Calculate scale for each row: scale = mean(abs(row_weights))
    // 2. Normalize row: scaled_row = row_weights / scale
    // 3. Ternarize: val > 0.5 -> 1, val < -0.5 -> -1, else 0
    // 4. Pack 4 ternary values into one i8 (2 bits per value)

    let mut ternary_values_temp = Array2::zeros(weights.raw_dim()); // Temp storage for {-1, 0, 1}

    for r in 0..num_rows {
        let row_view = weights.row(r);
        let mut current_scale = row_view.iter().map(|x| x.abs()).sum::<FloatType>() / num_cols as FloatType;
        if current_scale < 1e-9 { // Avoid division by zero or very small scales
            current_scale = 1e-9;
        }
        scales[r] = current_scale;

        for c in 0..num_cols {
            let scaled_val = weights[[r, c]] / current_scale;
            if scaled_val > 0.5 {
                ternary_values_temp[[r, c]] = 1;
            } else if scaled_val < -0.5 {
                ternary_values_temp[[r, c]] = -1;
            } else {
                ternary_values_temp[[r, c]] = 0;
            }
        }
    }

    // Pack ternary_values_temp into packed_weights
    // Mapping: 0 -> 00 (0), 1 -> 01 (1), -1 -> 10 (2)
    for r_idx in 0..num_rows {
        for c_packed_idx in 0.. (num_cols / 4) {
            let mut packed_byte: Int8Type = 0;
            for bit_pair_idx in 0..4 { // 4 pairs of 2 bits in a byte
                let c_ternary_idx = c_packed_idx * 4 + bit_pair_idx;
                let ternary_val = ternary_values_temp[[r_idx, c_ternary_idx]];

                let two_bit_val: Int8Type = match ternary_val {
                    1 => 1,  // 01
                    -1 => 2, // 10
                    _ => 0,  // 00 for 0
                };
                // Pack: LSB first value means first 2-bit value goes into bits 0,1 of the byte.
                packed_byte |= two_bit_val << (bit_pair_idx * 2);
            }
            packed_weights[[r_idx, c_packed_idx]] = packed_byte;
        }
    }
    (packed_weights, scales)
}

/// Quantizes `f32` activations to `i8` using per-tensor symmetric quantization.
///
/// This function takes a tensor of `f32` activations and quantizes them to `i8`
/// values in the range `[-127, 127]`. The quantization is symmetric, meaning
/// the zero point is mapped to zero.
///
/// The process involves:
/// 1. Finding the absolute maximum value (`abs_max`) in the input tensor.
/// 2. Calculating a single scaling factor for the entire tensor: `scale = abs_max / 127.0`.
///    If `abs_max` is zero, a scale of `1.0` is used to avoid division by zero.
/// 3. Quantizing each activation: `q_val = round(float_val / scale)`.
/// 4. Clamping the quantized values to the `i8` range `[-127, 127]`.
///
/// This approach is common for quantizing activations in models like BitNet, where
/// the dynamic range of activations is captured by a single scale factor.
///
/// # Arguments
/// * `activations`: A D-dimensional array view (`ArrayViewD<f32>`) of the `f32` activations.
///
/// # Returns
/// A tuple containing:
///   - `ArrayD<i8>`: The `i8` quantized activations, with the same shape as the input.
///   - `f32`: The scaling factor used for the quantization. This factor is needed for dequantization.
pub fn quantize_activations_int8_per_tensor(
    activations: &ArrayViewD<FloatType>
) -> (ArrayD<Int8Type>, FloatType) {
    if activations.is_empty() {
        return (ArrayD::from_shape_vec(activations.shape(), vec![]).unwrap(), 1.0);
    }

    // 1. Find the absolute maximum value in the tensor
    let abs_max = activations.iter().map(|&x| x.abs()).fold(FloatType::NEG_INFINITY, FloatType::max);

    // 2. Calculate the scale factor
    // Scale = max_abs_val / quant_max_val (e.g., 127 for int8)
    // We want to map f32 to [-127, 127] for i8
    // So, quantized_value = round(original_value / scale)
    // And original_value approx= quantized_value * scale
    // Thus, scale = abs_max / 127.0
    let scale = if abs_max == 0.0 {
        1.0 // Avoid division by zero if all activations are zero
    } else {
        abs_max / 127.0
    };

    // 3. Quantize the activations
    // Formula: q_val = round(float_val / scale).clip(-127, 127)
    // Note: The original BitLinear paper uses `s = 127 / input.abs().max()`, then `round(input * s)`.
    // This is equivalent to `round(input / (input.abs().max() / 127))`. So our scale is `abs_max / 127.0`.
    let mut quantized_activations = ArrayD::zeros(activations.shape());
    Zip::from(&mut quantized_activations)
        .and(activations)
        .for_each(|q_val, &float_val| {
            *q_val = (float_val / scale).round().max(-127.0).min(127.0) as Int8Type;
        });

    (quantized_activations, scale)
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr0, arr1, arr2, arr3, Array};

    #[test]
    fn test_quantize_activations_int8_per_tensor_basic() {
        let activations = arr2(&[[1.0, -2.0, 0.0], [127.0, -128.0, 60.0]]); // -128.0 will be clamped to -127 in symmetric
        let (q_acts, scale) = quantize_activations_int8_per_tensor(&activations.view());

        // abs_max is 128.0. Scale should be 128.0 / 127.0 approx 1.007874
        // Expected q_acts:
        // 1.0 / scale = 0.9921... -> round(0.9921) = 1
        // -2.0 / scale = -1.984... -> round(-1.984) = -2
        // 0.0 / scale = 0.0 -> round(0.0) = 0
        // 127.0 / scale = 125.9... -> round(126) = 126
        // -128.0 / scale = -126.9... -> round(-127) = -127
        // 60.0 / scale = 59.52... -> round(60) = 60

        let expected_q_acts = arr2(&[[1, -2, 0], [126, -127, 60]]);
        let expected_scale = 128.0 / 127.0;

        assert_eq!(q_acts, expected_q_acts);
        assert!((scale - expected_scale).abs() < 1e-6);

        // Test dequantization (approximate)
        let dequant_act = q_acts.mapv(|x| x as FloatType * scale);
        println!("Original: {:?}", activations);
        println!("Dequantized: {:?}", dequant_act);
        // Check that dequantized values are somewhat close to original, scaled by the quantization error
        // Example: 1.0 vs 1 * (128.0/127.0) = 1.007874
        //         -128.0 vs -127 * (128.0/127.0) = -128.0
        assert!((dequant_act[[0,0]] - activations[[0,0]]).abs() < scale); // error should be within scale
        assert!((dequant_act[[1,1]] - (-128.0_f32).max(-127.0*scale).min(127.0*scale)).abs() < scale * 1.5); // Clamped original
    }

    #[test]
    fn test_quantize_activations_int8_zeros() {
        let activations = Array::zeros((2,3)).into_dyn();
        let (q_acts, scale) = quantize_activations_int8_per_tensor(&activations.view());
        assert_eq!(q_acts, ArrayD::zeros(activations.shape()));
        assert_eq!(scale, 1.0); // Scale is 1.0 to avoid division by zero
    }

    #[test]
    fn test_quantize_activations_int8_single_value() {
        let activations = arr0(10.0).into_dyn(); // Scalar to Dyn
        let (q_acts, scale) = quantize_activations_int8_per_tensor(&activations.view());
        // abs_max = 10.0. scale = 10.0 / 127.0
        // q = round(10.0 / (10.0/127.0)) = round(127.0) = 127
        assert_eq!(q_acts.get(()).unwrap(), &127);
        assert!((scale - (10.0/127.0)).abs() < 1e-6);
    }

    #[test]
    fn test_quantize_activations_int8_empty() {
        let activations = ArrayD::<f32>::zeros(IxDyn(&[2,0,3]));
        let (q_acts, scale) = quantize_activations_int8_per_tensor(&activations.view());
        assert_eq!(q_acts.shape(), &[2,0,3]);
        assert_eq!(q_acts.len(), 0);
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn test_quantize_weights_ternary_placeholder() {
        // This is just to ensure the placeholder function runs without panic for valid inputs.
        let weights = arr2(&[[1.0, -2.0, 0.5, 0.1], [-0.1, 0.0, 3.0, -1.0]]);
        let (q_weights, scales) = quantize_weights_ternary(&weights.view());
        assert_eq!(q_weights.shape(), &[2, 1]); // Packed (4 cols -> 1 col)
        assert_eq!(scales.shape(), &[2]);
    }

     #[test]
    #[should_panic]
    fn test_quantize_weights_ternary_panic_cols_not_divisible_by_4() {
        let weights = Array2::<f32>::zeros((2,3));
        quantize_weights_ternary(&weights.view());
    }

    #[test]
    fn test_quantize_weights_ternary_empty_cols() {
        let weights = Array2::<f32>::zeros((2,0));
        let (q_weights, scales) = quantize_weights_ternary(&weights.view());
        assert_eq!(q_weights.shape(), &[2,0]);
        assert_eq!(scales.shape(), &[2]);
    }

    #[test]
    fn test_quantize_weights_ternary_logic() {
        // Row 1: Values will lead to 1, 0, -1, 1
        // abs_mean = (1.0 + 0.1 + 0.8 + 0.9) / 4 = 2.8 / 4 = 0.7
        // Scaled: 1.0/0.7=1.42 (->1), 0.1/0.7=0.14 (->0), -0.8/0.7=-1.14 (->-1), 0.9/0.7=1.28 (->1)
        // Ternary: [1, 0, -1, 1]
        // 2-bit:   [01, 00, 10, 01]
        // Packed (LSB first val): 01_10_00_01_bin = 0x61 = 97 (decimal i8)

        // Row 2: Values will lead to 0, -1, -1, 0
        // abs_mean = (0.2 + 0.7 + 0.6 + 0.3) / 4 = 1.8 / 4 = 0.45
        // Scaled: 0.2/0.45=0.44 (->0), -0.7/0.45=-1.55 (->-1), -0.6/0.45=-1.33 (->-1), 0.3/0.45=0.66 (->1, error in manual, should be 1)
        // Let's adjust second row for clearer ternaries:
        // Values: 0.2, -0.8, -0.7, 0.1
        // abs_mean = (0.2 + 0.8 + 0.7 + 0.1) / 4 = 1.8 / 4 = 0.45
        // Scaled: 0.2/0.45=0.44 (->0), -0.8/0.45=-1.77 (->-1), -0.7/0.45=-1.55 (->-1), 0.1/0.45=0.22 (->0)
        // Ternary: [0, -1, -1, 0]
        // 2-bit:   [00, 10, 10, 00]
        // Packed: 00_10_10_00_bin = 0x28 = 40 (decimal i8)

        let weights_f32 = arr2(&[
            [1.0, 0.1, -0.8, 0.9],  // Scale = 0.7. Ternary: [1,0,-1,1]. Packed: 01_10_00_01_bin = 97
            [0.2, -0.8, -0.7, 0.1]  // Scale = 0.45. Ternary: [0,-1,-1,0]. Packed: 00_10_10_00_bin = 40
        ]);

        let (packed, scales) = quantize_weights_ternary(&weights_f32.view());

        // Check scales
        assert!((scales[0] - 0.7).abs() < 1e-6);
        assert!((scales[1] - 0.45).abs() < 1e-6);

        // Check packed values
        // Expected packed:
        // val_1 = (1 << 0) | (2 << 2) | (0 << 4) | (1 << 6) = 1 | 8 | 0 | 64 = 73 (Incorrect manual calc: 01_10_00_01_bin = 1+0+8+64 = 73)
        // Let's re-verify packing:
        // Ternary: [1, 0, -1, 1] -> 2-bit [01, 00, 10, 01]
        // packed_byte = (01_base2) << 0 | (00_base2) << 2 | (10_base2) << 4 | (01_base2) << 6
        //             = 1 * (2^0)      | 0 * (2^2)      | 2 * (2^4)      | 1 * (2^6)
        //             = 1              | 0              | 2 * 16 = 32    | 1 * 64 = 64
        //             = 1 + 0 + 32 + 64 = 97. This matches the original calculation.

        // Ternary: [0, -1, -1, 0] -> 2-bit [00, 10, 10, 00]
        // packed_byte = (00_base2) << 0 | (10_base2) << 2 | (10_base2) << 4 | (00_base2) << 6
        //             = 0 * (2^0)      | 2 * (2^2)      | 2 * (2^4)      | 0 * (2^6)
        //             = 0              | 2 * 4 = 8      | 2 * 16 = 32    | 0
        //             = 0 + 8 + 32 + 0 = 40. This matches.

        assert_eq!(packed.shape(), &[2, 1]);
        assert_eq!(packed[[0,0]], 97i8);
        assert_eq!(packed[[1,0]], 40i8);
    }
     #[test]
    fn test_quantize_weights_ternary_all_zeros_row() {
        let weights_f32 = arr2(&[
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.1, -0.8, 0.9]
        ]);
        let (packed, scales) = quantize_weights_ternary(&weights_f32.view());

        assert!((scales[0] - 1e-9).abs() < 1e-12); // Scale for all zeros row becomes 1e-9
        assert_eq!(packed[[0,0]], 0i8); // All zeros packed is 0

        assert!((scales[1] - 0.7).abs() < 1e-6);
        assert_eq!(packed[[1,0]], 97i8);
    }

    #[test]
    fn test_quantize_weights_ternary_exact_thresholds() {
        // Values designed to hit thresholds for ternarization precisely
        // Scale for first row will be (0.5*0.7 + 0.0*0.7 + 0.5*0.7 + 0.0*0.7)/4 = (0.35+0.35)/4 = 0.7/4 = 0.175
        // [0.5*s, 0.0*s, -0.5*s, 0.4*s], where s = 0.175
        // Scaled: [0.5/0.175=2.85 (->1), 0 (->0), -0.5/0.175=-2.85 (->-1), 0.4/0.175=2.28 (->1)]
        // This needs careful construction if we want to test exact 0.5 boundaries
        // Let scale = 1 for simplicity of testing boundary conditions
        // Weights: [0.6, 0.2, -0.6, -0.2, 0.5, -0.5, 0.0, 1.0] (8 values for 2 packed bytes)
        // Scale = (0.6+0.2+0.6+0.2+0.5+0.5+0.0+1.0)/8 = 3.6/8 = 0.45
        let weights_f32_row = arr1(&[0.6, 0.2, -0.6, -0.2, 0.5, -0.5, 0.0, 1.0]);
        let scale_val = weights_f32_row.iter().map(|x|x.abs()).sum::<FloatType>() / weights_f32_row.len() as FloatType; // 0.45

        // Expected ternary based on scale 0.45:
        // 0.6/0.45 = 1.33  ->  1 (01)
        // 0.2/0.45 = 0.44  ->  0 (00)
        // -0.6/0.45= -1.33 -> -1 (10)
        // -0.2/0.45= -0.44 ->  0 (00)
        // Packed1: 00_10_00_01_bin = 32 + 1 = 33
        //
        // 0.5/0.45 = 1.11  ->  1 (01)
        // -0.5/0.45= -1.11 -> -1 (10)
        // 0.0/0.45 = 0.0   ->  0 (00)
        // 1.0/0.45 = 2.22  ->  1 (01)
        // Packed2: 01_00_10_01_bin = 64 + 8 + 1 = 73

        let weights_f32_mat = weights_f32_row.into_shape((1,8)).unwrap();
        let (packed, scales) = quantize_weights_ternary(&weights_f32_mat.view());

        assert!((scales[0] - scale_val).abs() < 1e-6);
        assert_eq!(packed.shape(), &[1,2]);
        assert_eq!(packed[[0,0]], 33i8);
        assert_eq!(packed[[0,1]], 73i8);
    }
}
