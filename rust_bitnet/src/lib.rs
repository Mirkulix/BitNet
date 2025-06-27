// Declare the modules that make up this crate

pub mod model;
pub mod layers;
pub mod quantization;
pub mod tensor_ops;

// Re-export key structs or functions if desired for easier access
// pub use model::BitNetModel;
// pub use layers::BitLinear;

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
