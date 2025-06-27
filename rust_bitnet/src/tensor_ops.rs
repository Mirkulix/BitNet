//! Provides specialized tensor operations, particularly for handling packed data formats.
//!
//! This module includes functions for unpacking quantized weights, which is a common
//! requirement in models that use low-bit quantization schemes like BitNet.

// Example: Function to unpack 2-bit weights from i8 into a new i8 array holding -1, 0, 1
use ndarray::{Array2, ArrayView2};

type Int8Type = i8;

/// Unpacks 2-bit weights, stored within `i8` values, back into their ternary `{-1, 0, 1}` representation.
///
/// Each `i8` element in the input `packed_weights` array is assumed to hold four 2-bit
/// ternary weights. The packing order is such that the first ternary value corresponds
/// to the least significant 2 bits of the byte.
///
/// The mapping from the 2-bit representation to ternary values is as follows:
/// - `00` (binary) -> `0`
/// - `01` (binary) -> `1`
/// - `10` (binary) -> `-1`
///
/// An invalid 2-bit pattern (`11` binary) will cause a panic, as it indicates an error
/// in the preceding quantization/packing stage.
///
/// # Arguments
/// * `packed_weights`: A 2D array view (`ArrayView2<i8>`) of the packed weights.
///   The number of columns in this array should be `original_ncols / 4`.
/// * `original_ncols`: The number of columns in the original, unpacked weight matrix.
///   This must be divisible by 4.
///
/// # Returns
/// * `Array2<i8>`: A new 2D array containing the unpacked ternary weights (values `-1`, `0`, or `1`).
///   The dimensions of the output array will be `(packed_weights.nrows(), original_ncols)`.
///
/// # Panics
/// - Panics if `original_ncols` is not divisible by 4 (implicitly, as `packed_weights.ncols()` would not match).
/// - Panics if `packed_weights.ncols()` is not equal to `original_ncols / 4`.
/// - Panics if an invalid 2-bit pattern (binary `11`) is encountered during unpacking.
pub fn unpack_i8_to_ternary(
    packed_weights: &ArrayView2<Int8Type>,
    original_ncols: usize
) -> Array2<Int8Type> {
    let num_rows = packed_weights.nrows();
    if original_ncols == 0 {
        return Array2::zeros((num_rows, 0));
    }
    // Packed dimension should be original_ncols / 4
    if packed_weights.ncols() != original_ncols / 4 {
        panic!(
            "Packed weights columns {} does not match original_ncols/4 ({}/{})",
            packed_weights.ncols(), original_ncols, 4
        );
    }

    let mut unpacked = Array2::zeros((num_rows, original_ncols));

    // Mapping from 2-bit value back to ternary:
    // 0 (00_bin) ->  0
    // 1 (01_bin) ->  1
    // 2 (10_bin) -> -1
    // 3 (11_bin) -> undefined/error (should not occur with correct packing)
    for r_idx in 0..num_rows {
        for c_packed_idx in 0..packed_weights.ncols() {
            let packed_byte = packed_weights[[r_idx, c_packed_idx]];
            for bit_pair_idx in 0..4 { // 4 pairs of 2 bits in a byte
                let two_bit_val = (packed_byte >> (bit_pair_idx * 2)) & 0x03; // Extract 2 bits
                let c_unpacked_idx = c_packed_idx * 4 + bit_pair_idx;

                let ternary_val: Int8Type = match two_bit_val {
                    0 => 0,
                    1 => 1,
                    2 => -1,
                    3 => {
                        // This case should ideally not happen if packing is done correctly.
                        // Handle as error or a default value. For now, panic.
                        panic!("Invalid 2-bit value '3' encountered during unpacking at row {}, col_packed {}, pair_idx {}. Packed byte: {:02x}",
                               r_idx, c_packed_idx, bit_pair_idx, packed_byte);
                    }
                    _ => unreachable!(), // Should not happen with & 0x03
                };
                unpacked[[r_idx, c_unpacked_idx]] = ternary_val;
            }
        }
    }
    unpacked
}

// Other specific operations can be added here, for example:
// - Specialized matrix multiplication for i8 * i8 if ndarray's dot is not optimal
//   or if specific accumulation (e.g. to i32) is needed before scaling.
// - Operations that handle the KVCache for attention.

use anyhow::{Context, Result};
use base64::{Engine as _, engine::general_purpose::STANDARD as base64_standard};

/// Decodes a Base64 encoded string into a vector of bytes.
///
/// # Arguments
/// * `b64_string`: The Base64 encoded string.
///
/// # Returns
/// * `Result<Vec<u8>>`: A vector of bytes if decoding is successful.
pub fn decode_base64(b64_string: &str) -> Result<Vec<u8>> {
    base64_standard.decode(b64_string).context("Failed to decode Base64 string")
}

/// Reconstructs an ndarray::Array<T, D> from a byte vector and shape information.
///
/// This function assumes the byte vector represents tightly packed data of type `T`.
/// The number of bytes must be consistent with the anzahl of elements and `std::mem::size_of::<T>()`.
///
/// # Type Parameters
/// * `T`: The data type of the array elements (e.g., `f32`, `i8`). Must implement `Clone` and `Default`.
/// * `D`: The dimensionality of the array (e.g., `Ix1` for `Array1`, `Ix2` for `Array2`).
///
/// # Arguments
/// * `bytes`: A vector of bytes containing the raw tensor data.
/// * `shape`: A slice representing the desired shape of the array.
///
/// # Returns
/// * `Result<ndarray::Array<T, D>>`: The reconstructed ndarray Array.
///
/// # Panics
/// Panics if the number of bytes does not match the expected size for the given shape and data type.
pub fn array_from_bytes<T, D>(bytes: Vec<u8>, shape: &[usize]) -> Result<ndarray::Array<T, D>>
where
    T: Clone + Default + Copy, // Added Copy as T is likely a primitive
    D: ndarray::Dimension,
{
    let elem_size = std.mem::size_of::<T>();
    let expected_elements = shape.iter().product::<usize>();
    let expected_bytes = expected_elements * elem_size;

    if bytes.len() != expected_bytes {
        return Err(anyhow::anyhow!(
            "Byte vector length {} does not match expected size {} for shape {:?} and type_size {}",
            bytes.len(), expected_bytes, shape, elem_size
        ));
    }

    // Create an uninitialized array of the correct shape.
    // Using `unsafe` here is generally discouraged unless absolutely necessary and well-understood.
    // A safer approach would be to initialize with default values and then fill,
    // or use a method that directly constructs from a slice of T if available.
    // For primitives, direct casting from byte slices is possible but needs careful alignment handling.

    // Simpler and safer way for primitives:
    let mut data_vec: Vec<T> = Vec::with_capacity(expected_elements);
    // This assumes T is a primitive type where direct casting from chunks of bytes is safe.
    // This is generally true for f32, i8, etc., on platforms with matching endianness.
    // The conversion script should ensure data is saved in native endianness or a fixed one.
    unsafe {
        data_vec.set_len(expected_elements); // Initialize vec length
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), data_vec.as_mut_ptr() as *mut u8, bytes.len());
    }

    ndarray::Array::from_shape_vec(D::from_dimension(&ndarray::IxDyn(shape)).context("Invalid shape for ndarray")?, data_vec)
        .context("Failed to create ndarray from shape and vector")
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, Array, Ix1, Ix2}; // Added Ix1, Ix2

    #[test]
    fn test_unpack_i8_to_ternary_basic() {
        // Packed values from test_quantize_weights_ternary_logic:
        // Row 1: Ternary [1, 0, -1, 1] -> Packed 97i8 (01100001_bin)
        // Row 2: Ternary [0, -1, -1, 0] -> Packed 40i8 (00101000_bin)
        let packed_data = arr2(&[[97i8], [40i8]]);
        let original_ncols = 4;

        let unpacked = unpack_i8_to_ternary(&packed_data.view(), original_ncols);

        assert_eq!(unpacked.shape(), &[2, 4]);
        assert_eq!(unpacked.row(0), arr2(&[[1, 0, -1, 1]]).into_dyn().into_shape((4,)).unwrap());
        assert_eq!(unpacked.row(1), arr2(&[[0, -1, -1, 0]]).into_dyn().into_shape((4,)).unwrap());
    }

    #[test]
    fn test_unpack_i8_to_ternary_multiple_packed_cols() {
        // Packed1: 33i8 from [0.6, 0.2, -0.6, -0.2] (scale 0.45) -> Ternary [1,0,-1,0]
        // Packed2: 73i8 from [0.5, -0.5, 0.0, 1.0] (scale 0.45) -> Ternary [1,-1,0,1]
        let packed_data = arr2(&[[33i8, 73i8]]);
        let original_ncols = 8;
        let unpacked = unpack_i8_to_ternary(&packed_data.view(), original_ncols);

        assert_eq!(unpacked.shape(), &[1, 8]);
        let expected_row0 = arr2(&[[1, 0, -1, 0, 1, -1, 0, 1]]);
        assert_eq!(unpacked.row(0), expected_row0.into_dyn().into_shape((8,)).unwrap());
    }

    #[test]
    fn test_unpack_i8_to_ternary_all_zeros() {
        let packed_data = arr2(&[[0i8, 0i8]]);
        let original_ncols = 8;
        let unpacked = unpack_i8_to_ternary(&packed_data.view(), original_ncols);
        assert_eq!(unpacked, Array2::zeros((1,8)));
    }

    #[test]
    fn test_unpack_i8_to_ternary_all_ones() {
        // Ternary [1,1,1,1] -> 2-bit [01,01,01,01] -> Packed 01010101_bin = 85
        let packed_data = arr2(&[[85i8]]);
        let original_ncols = 4;
        let unpacked = unpack_i8_to_ternary(&packed_data.view(), original_ncols);
        assert_eq!(unpacked, arr2(&[[1,1,1,1]]));
    }

    #[test]
    fn test_unpack_i8_to_ternary_all_minus_ones() {
        // Ternary [-1,-1,-1,-1] -> 2-bit [10,10,10,10] -> Packed 10101010_bin = -86 (or 170 as u8)
        let packed_val = -86i8;
        let packed_data = arr2(&[[packed_val]]);
        let original_ncols = 4;
        let unpacked = unpack_i8_to_ternary(&packed_data.view(), original_ncols);
        assert_eq!(unpacked, arr2(&[[-1,-1,-1,-1]]));
    }

    #[test]
    #[should_panic(expected = "Invalid 2-bit value '3' encountered during unpacking")]
    fn test_unpack_i8_to_ternary_invalid_pattern() {
        let packed_data = arr2(&[[-64i8]]); // -64 is 11000000_bin, first 2-bit is 00, second is 00, third is 00, fourth is 11.
        unpack_i8_to_ternary(&packed_data.view(), 4);
    }

    #[test]
    #[should_panic(expected = "Packed weights columns 1 does not match original_ncols/4 (3/4)")]
    fn test_unpack_i8_to_ternary_shape_mismatch() {
        let packed_data = arr2(&[[1i8]]);
        unpack_i8_to_ternary(&packed_data.view(), 3);
    }

    #[test]
    fn test_unpack_i8_to_ternary_empty_original_cols() {
        let packed_data = Array2::<i8>::zeros((2,0));
        let unpacked = unpack_i8_to_ternary(&packed_data.view(), 0);
        assert_eq!(unpacked.shape(), &[2,0]);
    }

    // Tests for new helper functions
    #[test]
    fn test_decode_base64_valid() {
        // "hello" -> aGVsbG8=
        let b64_str = "aGVsbG8=";
        let bytes = decode_base64(b64_str).unwrap();
        assert_eq!(bytes, b"hello");
    }

    #[test]
    fn test_decode_base64_invalid() {
        let b64_str = "invalid-b64-string";
        assert!(decode_base64(b64_str).is_err());
    }

    #[test]
    fn test_array_from_bytes_f32_1d() {
        let data_f32: Vec<f32> = vec![1.0, 2.5, -3.0];
        let mut bytes_vec: Vec<u8> = Vec::new();
        for val in &data_f32 {
            bytes_vec.extend_from_slice(&val.to_ne_bytes());
        }

        let shape = [3];
        let array: Array<f32, Ix1> = array_from_bytes(bytes_vec, &shape).unwrap();
        assert_eq!(array, arr1(&data_f32));
    }

    #[test]
    fn test_array_from_bytes_i8_2d() {
        let data_i8: Vec<i8> = vec![1, -2, 3, 4, -5, 6];
        let bytes_vec: Vec<u8> = data_i8.iter().map(|&x| x as u8).collect();

        let shape = [2,3];
        let array: Array<i8, Ix2> = array_from_bytes(bytes_vec, &shape).unwrap();
        let expected_array = arr2(&[[1, -2, 3], [4, -5, 6]]);
        assert_eq!(array, expected_array);
    }

    #[test]
    fn test_array_from_bytes_wrong_size() {
        let bytes_vec: Vec<u8> = vec![0,0,0,0]; // 4 bytes for one f32
        let shape = [2]; // Expects 2 f32s (8 bytes)
        let result: Result<Array<f32, Ix1>> = array_from_bytes(bytes_vec, &shape);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Byte vector length 4 does not match expected size 8"));
    }
     #[test]
    fn test_array_from_bytes_empty() {
        let bytes_vec: Vec<u8> = vec![];
        let shape = [0];
        let array: Array<f32, Ix1> = array_from_bytes(bytes_vec.clone(), &shape).unwrap();
        assert_eq!(array.len(), 0);

        let shape2d = [2,0];
        let array2d: Array<f32, Ix2> = array_from_bytes(bytes_vec, &shape2d).unwrap();
        assert_eq!(array2d.shape(), &[2,0]);
    }
}
