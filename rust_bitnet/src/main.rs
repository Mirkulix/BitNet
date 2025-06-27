use clap::Parser;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use anyhow::{Result, Context}; // Using anyhow for error handling

// Assuming these will be the primary structs from our library
// We'll need to deserialize the JSON into something Rust can use.
// This might involve creating specific structs for deserialization that mirror the JSON structure.
use rust_bitnet::model::{BitNetModel, ModelArgs}; // Assuming ModelArgs is also needed for loading
// use rust_bitnet::layers::{BitLinear, Embedding, RMSNorm}; // If individual layer loading is needed

// --- Structs for deserializing weights from JSON ---
// These need to match the structure created by convert_weights.py
use serde::Deserialize;
use ndarray::{Array1, Array2}; // For actual tensor types after decoding

#[derive(Deserialize, Debug)]
struct JsonModelArgs {
    dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: Option<usize>,
    vocab_size: usize,
    ffn_dim_multiplier: Option<f64>,
    ffn_hidden_dim: Option<usize>, // Made Option to handle potential null from JSON if not calculated
    norm_eps: f32,
    rope_theta: Option<f32>,
}

impl From<JsonModelArgs> for ModelArgs {
    fn from(json_args: JsonModelArgs) -> Self {
        ModelArgs {
            dim: json_args.dim,
            n_layers: json_args.n_layers,
            n_heads: json_args.n_heads,
            n_kv_heads: json_args.n_kv_heads,
            vocab_size: json_args.vocab_size,
            ffn_dim_multiplier: json_args.ffn_dim_multiplier,
            ffn_hidden_dim: json_args.ffn_hidden_dim,
            norm_eps: json_args.norm_eps,
            rope_theta: json_args.rope_theta,
        }
    }
}


#[derive(Deserialize, Debug)]
struct JsonBitLinearWeights {
    packed_weights_b64: String,
    scales_b64: String,
    original_shape: Vec<usize>,
    packed_shape: Vec<usize>,
    scales_shape: Vec<usize>,
    // dtype_packed: String, // We'll assume i8
    // dtype_scales: String, // We'll assume f32
    is_bitlinear: bool,
}

#[derive(Deserialize, Debug)]
struct JsonF32Weights {
    weights_b64: String,
    shape: Vec<usize>,
    // dtype: String, // We'll assume f32
    is_bitlinear: bool,
}

#[derive(Deserialize, Debug)]
#[serde(untagged)] // Allows deserializing into either BitLinear or F32 variant based on fields
enum JsonWeightVariant {
    BitLinear(JsonBitLinearWeights),
    F32(JsonF32Weights),
}

#[derive(Deserialize, Debug)]
struct ConvertedModelData {
    model_args: JsonModelArgs,
    weights: std::collections::HashMap<String, JsonWeightVariant>,
}


/// Command Line Interface for BitNet Rust Inference
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct CliArgs {
    /// Path to the converted BitNet model JSON file
    #[clap(short, long, value_parser)]
    model_path: PathBuf,

    /// Input prompt for text generation
    #[clap(short, long, value_parser)]
    prompt: String,

    /// Maximum number of new tokens to generate
    #[clap(short, long, value_parser, default_value_t = 50)]
    max_tokens: usize,

    /// Temperature for sampling (e.g., 0.7)
    #[clap(short, long, value_parser, default_value_t = 0.7)]
    temperature: f32,
    // TODO: Add other generation parameters like top_p, top_k if needed
}

fn load_model_from_json(path: &PathBuf) -> Result<BitNetModel> {
    println!("Loading model from: {:?}", path);
    let mut file = File::open(path).context(format!("Failed to open model file: {:?}", path))?;
    let mut contents = String::new();
    file.read_to_string(&mut contents).context("Failed to read model file contents")?;

    let data: ConvertedModelData = serde_json::from_str(&contents).context("Failed to parse model JSON")?;

    let rust_model_args: ModelArgs = data.model_args.into();
    println!("Model args loaded: {:?}", rust_model_args);

    let mut model = BitNetModel::new(rust_model_args.clone()); // Assuming new initializes with empty/default weights

    // TODO: Iterate through data.weights and load them into the `model`
    // This will involve:
    // 1. Matching keys from JSON to layers/parameters in `BitNetModel` struct.
    // 2. Base64 decoding the weight strings.
    // 3. Reconstructing ndarray::Array from bytes and shape information.
    // 4. Calling appropriate `load_weights` methods on the model's layers.
    //    (e.g., model.tok_embeddings.load_weights(...), model.layers[i].attention_norm.load_weights(...), etc.)

    for (key, weight_variant) in data.weights {
        println!("Processing weight key: {}", key);
        match weight_variant {
            JsonWeightVariant::BitLinear(w) => {
                // Decode packed_weights_b64 and scales_b64
                // Reconstruct Array2<i8> and Array1<f32>
                // Find the corresponding BitLinear layer in `model` and call `load_weights`
                println!("  BitLinear weights: packed_shape {:?}, scales_shape {:?}", w.packed_shape, w.scales_shape);
                // Example:
                // let packed_bytes = base64::decode(&w.packed_weights_b64).context("Failed to decode packed_weights_b64")?;
                // let scales_bytes = base64::decode(&w.scales_b64).context("Failed to decode scales_b64")?;
                // ... then reconstruct ndarrays ...
            }
            JsonWeightVariant::F32(w) => {
                // Decode weights_b64
                // Reconstruct Array (usually Array2<f32> for embeddings, Array1<f32> for norms)
                // Find the corresponding layer in `model` and call `load_weights`
                println!("  F32 weights: shape {:?}", w.shape);
            }
        }
    }

    println!("Model structure initialized. Weight loading logic is a TODO.");
    Ok(model)
}

fn generate_text(
    model: &BitNetModel,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
) -> Result<String> {
    println!("\nGenerating text for prompt: \"{}\"", prompt);
    println!("Max new tokens: {}, Temperature: {}", max_tokens, temperature);

    // TODO:
    // 1. Initialize Tokenizer (this is a major missing piece)
    //    - Need a tokenizer model file (e.g., from Hugging Face, SentencePiece .model)
    //    - Use a Rust tokenizer crate (e.g., `tokenizers`)
    // let tokenizer = load_tokenizer("path/to/tokenizer.model")?;

    // 2. Tokenize the input prompt
    // let input_tokens = tokenizer.encode(prompt, true).map_err(|e| anyhow::anyhow!("Tokenization error: {}",e))?.get_ids();
    // let mut current_tokens: Vec<usize> = input_tokens.iter().map(|&x| x as usize).collect();
    let mut current_tokens: Vec<usize> = prompt.chars().map(|c| c as usize % model.args.vocab_size).collect(); // Dummy tokenizer
    println!("Dummy tokenized input: {:?}", current_tokens);


    let mut generated_sequence = current_tokens.clone();
    let mut new_tokens_generated = 0;

    for _i in 0..max_tokens {
        // Prepare input for the model (e.g., ArrayView1<usize>)
        let model_input = ndarray::Array::from_vec(current_tokens.clone());

        // Get logits from the model
        // This currently returns dummy zeros of shape (current_tokens.len(), vocab_size)
        // We need logits for the *next* token, so typically we take the logits of the *last* token in the sequence.
        let logits_all_tokens = model.forward(&model_input.view()); // This needs to handle KVCache internally for efficiency

        // Get logits for the last token: (vocab_size,)
        let next_token_logits = logits_all_tokens.row(logits_all_tokens.nrows() - 1);

        // TODO: Implement sampling (e.g., temperature sampling, top-k, top-p)
        // For now, argmax (greedy decoding) as a placeholder
        let mut next_token_id = 0;
        let mut max_logit = -f32::INFINITY;
        for (id, &logit_val) in next_token_logits.iter().enumerate() {
            if logit_val > max_logit {
                max_logit = logit_val;
                next_token_id = id;
            }
        }

        // TODO: Handle End-Of-Sequence (EOS) token
        // if next_token_id == tokenizer.eos_token_id() { break; }

        current_tokens.push(next_token_id);
        generated_sequence.push(next_token_id);
        new_tokens_generated += 1;

        // Optional: Print the generated token (or its string representation)
        // print!("{}", tokenizer.decode(&[next_token_id as u32], false).unwrap_or("?".to_string()));
        // std::io::stdout().flush().unwrap();

        if new_tokens_generated >= max_tokens {
            break;
        }
    }
    println!("\nFinished generation.");

    // TODO: Decode the full generated_sequence back to text using the tokenizer
    // let output_text = tokenizer.decode(generated_sequence.iter().map(|&x| x as u32).collect(), true)
    //    .map_err(|e| anyhow::anyhow!("Decoding error: {}", e))?;

    let output_text = generated_sequence.iter().map(|&id| std::char::from_u32(id as u32 % 255 + 32).unwrap_or('?')).collect::<String>(); // Dummy detokenizer
    Ok(output_text)
}


fn main() -> Result<()> {
    let cli_args = CliArgs::parse();

    println!("CLI Args: {:?}", cli_args);

    // 1. Load the model (weights are placeholders for now)
    let model = load_model_from_json(&cli_args.model_path)
        .context("Failed to load model")?;

    // 2. Generate text
    let generated_text = generate_text(
        &model,
        &cli_args.prompt,
        cli_args.max_tokens,
        cli_args.temperature,
    ).context("Failed during text generation")?;

    println!("\n--- Generated Text ---");
    println!("{}", generated_text);
    println!("--- End of Text ---");

    Ok(())
}
