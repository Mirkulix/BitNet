import torch
import json
import base64
import argparse
import numpy as np
from typing import Dict, Any, Tuple

# Placeholder for the actual model definition if needed for direct loading.
# Often, we just load a state_dict and map keys.
# from somewhere import BitNetModelPyTorch, ModelArgsPyTorch # Hypothetical Python model definition

# --- Quantization specific to BitNet ---
# This needs to align with the Rust implementation's expectations.

def quantize_weights_ternary_python(
    weights_fp32: np.ndarray,
    threshold: float = 0.05 # Example threshold for determining zero value, or use 1.58-bit specific logic
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantizes fp32 weights to ternary {-1, 0, 1}, then packs them into 2-bit representations within an int8 array.
    Also returns the scaling factor for the weights.

    This is a simplified version. The actual 1.58-bit quantization might be more complex,
    involving absmean scaling first, then ternary decision.
    For BitNet (1.58-bit), the process is typically:
    1. Scale weights: W_scaled = W / absmean(W) (or other scaling factor like absmax)
    2. Ternarize W_scaled to {-1, 0, 1} based on thresholds (e.g. > tau -> 1, < -tau -> -1, else 0)
       Or, for strict 1.58-bit: round to nearest {-1, 0, 1} after scaling.
       The original "BitNet: Scaling 1-bit Transformers for Large Language Models" paper suggests:
       W_scaled = W / mean(abs(W_fp32))
       W_quant = round_to_nearest(W_scaled, {-1, 0, 1}) (or a specific rounding function)

    Here, we'll implement a more direct ternary mapping based on values and then pack.
    The scale returned will be the absmax of the original weights per output channel,
    which is a common way to dequantize.

    Args:
        weights_fp32 (np.ndarray): Input f32 weights (out_features, in_features).
        threshold (float): A value (relative to scale) to determine the zero region. Not used in strict BitNet absmean.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - packed_i8_weights (np.ndarray): Shape (out_features, in_features // 4), dtype int8.
            - scales_fp32 (np.ndarray): Shape (out_features,), dtype float32. Scales are absmax of original weights.
    """
    out_features, in_features = weights_fp32.shape
    if in_features % 4 != 0:
        raise ValueError(f"in_features ({in_features}) must be divisible by 4 for 2-bit packing.")

    # Calculate scales (absmax per output channel of the original weights)
    scales_fp32 = np.max(np.abs(weights_fp32), axis=1).astype(np.float32)
    scales_fp32[scales_fp32 == 0] = 1.0 # Avoid division by zero if a whole row is zero

    ternary_weights = np.zeros_like(weights_fp32, dtype=np.int8)

    for r in range(out_features):
        # This is a simplified ternarization for demonstration.
        # Proper BitNet 1.58 bit quantization uses specific scaling and rounding.
        # W_beta = W / (mean(abs(W)) + eps)
        # W_b = round_to_nearest(W_beta, {-1,0,1})
        # For this example, let's use a simple sign-based ternarization for non-zero elements
        # scaled by the row's absmax (which is our scale).
        # This is NOT the BitNet 1.58 bit method but a placeholder.
        # A more accurate one would involve global absmean scaling or per-tensor scaling then ternarization.
        # Let's assume weights are already appropriately scaled if we were to use a fixed threshold.
        # For now, just map to -1, 0, 1 based on sign for values far from zero.
        # A simple (but not BitNet's) approach:
        # threshold_val = threshold * scales_fp32[r] # if threshold was relative
        # ternary_weights[r, weights_fp32[r,:] > threshold_val] = 1
        # ternary_weights[r, weights_fp32[r,:] < -threshold_val] = -1

        # Let's try to mimic the BitNet paper's spirit more closely:
        # Step 1: Calculate mean of absolute values for the row (or global, depending on paper interpretation for W_q)
        row_abs_mean = np.mean(np.abs(weights_fp32[r, :]))
        if row_abs_mean == 0: row_abs_mean = 1e-9 # Avoid division by zero

        scaled_row = weights_fp32[r, :] / row_abs_mean

        # Step 2: Round to nearest {-1, 0, 1}
        # Simplified rounding: if val > 0.5 -> 1, if val < -0.5 -> -1, else 0
        # This threshold (0.5) is common for rounding.
        ternary_weights[r, scaled_row > 0.5] = 1
        ternary_weights[r, scaled_row < -0.5] = -1
        # All others remain 0 due to np.zeros_like initialization.

    # Pack ternary_weights (-1, 0, 1) into 2-bit representation.
    # Mapping:
    #  0  -> 00 (binary) -> 0 (decimal)
    #  1  -> 01 (binary) -> 1 (decimal)
    # -1  -> 10 (binary) -> 2 (decimal)
    # (Or another mapping, this must be consistent with Rust's unpack)
    packed_i8_weights = np.zeros((out_features, in_features // 4), dtype=np.int8)
    for r in range(out_features):
        for c_packed in range(in_features // 4):
            val_packed = 0
            for bit_idx in range(4):
                c_ternary = c_packed * 4 + bit_idx
                ternary_val = ternary_weights[r, c_ternary]

                two_bit_val = 0
                if ternary_val == 1:
                    two_bit_val = 1
                elif ternary_val == -1:
                    two_bit_val = 2
                # if ternary_val == 0, two_bit_val remains 0

                val_packed |= (two_bit_val << (bit_idx * 2)) # Pack from left-to-right in byte (LSB first value)
            packed_i8_weights[r, c_packed] = val_packed

    return packed_i8_weights, scales_fp32


def convert_model_to_rust_format(
    pytorch_model_path: str,
    output_json_path: str,
    model_args_override: Dict[str, Any] = None
):
    """
    Converts a PyTorch model checkpoint to a JSON format readable by the Rust application.
    """
    try:
        # Try to load as safetensors first, then fallback to .pth
        from safetensors.torch import load_file
        state_dict = load_file(pytorch_model_path, device="cpu")
        print(f"Successfully loaded model from {pytorch_model_path} (safetensors).")
    except Exception:
        try:
            state_dict = torch.load(pytorch_model_path, map_location="cpu")
            # If it's a full model, get the state_dict
            if hasattr(state_dict, 'state_dict'):
                state_dict = state_dict.state_dict()
            elif not isinstance(state_dict, dict):
                raise ValueError("Loaded object is not a state_dict or model.")
            print(f"Successfully loaded model from {pytorch_model_path} (pth).")
        except Exception as e:
            print(f"Could not load model from {pytorch_model_path}. Error: {e}")
            return

    converted_data: Dict[str, Any] = {
        "model_args": {}, # Will be populated
        "weights": {}
    }

    # --- Populate ModelArgs ---
    # This is crucial. The Rust ModelArgs struct needs to be mirrored here.
    # These might come from a config file associated with the PyTorch model,
    # or need to be manually specified if not in the checkpoint.
    # For demonstration, using some defaults that might be overridden.
    # Ideally, the PyTorch model checkpoint would also store its config.
    # If your PyTorch model has a `config` attribute (like Hugging Face models):
    # if hasattr(loaded_model_or_state_dict, 'config'):
    #    pt_config = loaded_model_or_state_dict.config
    #    args = ModelArgsPyTorch.from_pt_config(pt_config) # hypothetical
    # else:
    # For now, we'll assume ModelArgs are partially hardcoded or passed via model_args_override
    # This needs to match the Rust ModelArgs struct definition!
    # Keys: dim, n_layers, n_heads, n_kv_heads, vocab_size, ffn_hidden_dim, norm_eps etc.
    # Default values that should match Rust's ModelArgs::default() or be configurable.
    default_args = {
        "dim": 2048, "n_layers": 24, "n_heads": 32, "n_kv_heads": None,
        "vocab_size": 32000, "ffn_hidden_dim": None, "ffn_dim_multiplier": (2.0/3.0*4.0),
        "norm_eps": 1e-5, "rope_theta": 10000.0
    }
    if model_args_override:
        default_args.update(model_args_override)

    # Calculate ffn_hidden_dim if multiplier is given and hidden_dim is not
    if default_args.get("ffn_hidden_dim") is None and default_args.get("ffn_dim_multiplier") is not None:
        dim = default_args["dim"]
        multiplier = default_args["ffn_dim_multiplier"]
        # Standard FFN hidden dim calculation (e.g., for SwiGLU)
        # hidden_dim = 4 * dim
        # hidden_dim = int(2 * hidden_dim / 3)
        # hidden_dim = (hidden_dim + 7) // 8 * 8 # Multiple of 8
        # This calculation needs to be precise as per the model's design.
        # The python code has `ffn_dim = 6912` for `dim = 2560`.
        # `(4 * 2560 * 2/3) = 6826.66`.
        # `(4 * args.dim * 2 / 3 + args.multiple_of -1) // args.multiple_of * args.multiple_of`
        # For now, let's use a simplified calculation or expect it to be provided.
        # If ffn_dim_multiplier = (2/3 * 4), then hidden_dim = dim * (8/3)
        ffn_hidden_dim = int(dim * multiplier)
        default_args["ffn_hidden_dim"] = ffn_hidden_dim
        print(f"Calculated ffn_hidden_dim: {ffn_hidden_dim} from dim: {dim} and multiplier: {multiplier}")


    converted_data["model_args"] = default_args
    print(f"Using ModelArgs: {converted_data['model_args']}")


    # --- Weight Conversion ---
    # The keys in state_dict need to be mapped to the keys expected by the Rust model.
    # This mapping is CRITICAL and model-specific.
    # Example mapping (highly dependent on your PyTorch model's layer naming):
    key_map = {
        # PyTorch key : Rust key pattern (can use format strings for layer numbers)
        "tok_embeddings.weight": "tok_embeddings.weight",
        "norm.weight": "norm.weight", # Final RMSNorm
        "output.weight": "output.weight", # LM Head
        # For Transformer Blocks (e.g., Llama-style naming)
        "layers.{}.attention.wq.weight": "layers.{}.attention.wq.weight",
        "layers.{}.attention.wk.weight": "layers.{}.attention.wk.weight",
        "layers.{}.attention.wv.weight": "layers.{}.attention.wv.weight",
        "layers.{}.attention.wo.weight": "layers.{}.attention.wo.weight",
        "layers.{}.feed_forward.w1.weight": "layers.{}.feed_forward.w1.weight", # Or w13 for SwiGLU
        "layers.{}.feed_forward.w2.weight": "layers.{}.feed_forward.w2.weight",
        "layers.{}.feed_forward.w3.weight": "layers.{}.feed_forward.w3.weight", # If SwiGLU
        "layers.{}.attention_norm.weight": "layers.{}.attention_norm.weight",
        "layers.{}.ffn_norm.weight": "layers.{}.ffn_norm.weight",
    }

    processed_rust_keys = set()

    for pt_key, tensor_fp32 in state_dict.items():
        tensor_fp32_np = tensor_fp32.to(torch.float32).numpy()
        rust_key = None

        # Direct mapping for non-layer-specific keys
        if pt_key in key_map:
            rust_key = key_map[pt_key]
        else:
            # Attempt to map layer-specific keys
            parts = pt_key.split('.')
            if len(parts) > 2 and parts[0] == 'layers' and parts[1].isdigit():
                layer_idx = int(parts[1])
                pt_pattern_suffix = ".".join(parts[2:])
                # Find a pattern like "layers.{}.attention_norm.weight"
                # We need to reconstruct the PT pattern to search in key_map values.
                # This is a bit simplified. A more robust solution would iterate key_map.
                # For now, assume a direct suffix match after "layers.{layer_idx}."

                # Let's try to find a suitable key_map entry
                # (e.g. "attention.wq.weight" if pt_key_suffix is "attention.wq.weight")
                pt_key_suffix_for_lookup = pt_pattern_suffix

                found_pattern = None
                for k_pt, k_rust_pattern in key_map.items():
                    if k_pt.startswith("layers.{}.") and k_pt.endswith(pt_key_suffix_for_lookup):
                        found_pattern = k_rust_pattern
                        break

                if found_pattern:
                    rust_key = found_pattern.format(layer_idx)

        if rust_key is None:
            print(f"Skipping unmatched PyTorch key: {pt_key}")
            continue

        if rust_key in processed_rust_keys:
            print(f"Warning: Rust key {rust_key} (from PT key {pt_key}) has already been processed. Check key_map for duplicates.")
            continue

        print(f"Processing: {pt_key} -> {rust_key} with shape {tensor_fp32_np.shape}")

        # Determine if it's a BitLinear weight based on the rust_key pattern
        # This is an approximation; a more robust way is to know which layers are BitLinear.
        is_bitlinear_weight = ".wq." in rust_key or \
                              ".wk." in rust_key or \
                              ".wv." in rust_key or \
                              ".wo." in rust_key or \
                              ".feed_forward.w1." in rust_key or \
                              ".feed_forward.w2." in rust_key or \
                              ".feed_forward.w3." in rust_key or \
                              rust_key == "output.weight" # Assuming LM head is also BitLinear

        if is_bitlinear_weight and "weight" in rust_key: # Ensure it's a weight tensor
            packed_i8_weights, scales_fp32 = quantize_weights_ternary_python(tensor_fp32_np)

            converted_data["weights"][rust_key] = {
                "packed_weights_b64": base64.b64encode(packed_i8_weights.tobytes()).decode('utf-8'),
                "scales_b64": base64.b64encode(scales_fp32.tobytes()).decode('utf-8'),
                "original_shape": list(tensor_fp32_np.shape), # For verification
                "packed_shape": list(packed_i8_weights.shape),
                "scales_shape": list(scales_fp32.shape),
                "dtype_packed": str(packed_i8_weights.dtype),
                "dtype_scales": str(scales_fp32.dtype),
                "is_bitlinear": True
            }
            print(f"  Quantized and packed {rust_key}.")
        elif "weight" in rust_key: # For non-BitLinear weights (e.g., embeddings, RMSNorm)
            converted_data["weights"][rust_key] = {
                "weights_b64": base64.b64encode(tensor_fp32_np.tobytes()).decode('utf-8'),
                "shape": list(tensor_fp32_np.shape),
                "dtype": str(tensor_fp32_np.dtype),
                "is_bitlinear": False
            }
            print(f"  Stored {rust_key} as f32.")
        else:
            print(f"  Skipping non-weight tensor: {rust_key}")
            continue

        processed_rust_keys.add(rust_key)


    # --- Sanity Checks ---
    # Check if all expected keys for the given n_layers are present
    n_layers_from_args = converted_data["model_args"]["n_layers"]
    expected_layer_keys = [
        pattern for pt_pattern, pattern in key_map.items() if pt_pattern.startswith("layers.{}.")
    ]
    missing_keys = False
    for i in range(n_layers_from_args):
        for pattern in expected_layer_keys:
            expected_key = pattern.format(i)
            if expected_key not in processed_rust_keys:
                print(f"Warning: Expected Rust key {expected_key} for layer {i} not found in processed weights.")
                missing_keys = True
    if not missing_keys:
        print("All expected per-layer keys seem to be processed.")


    # --- Save to JSON ---
    try:
        with open(output_json_path, 'w') as f:
            json.dump(converted_data, f, indent=4)
        print(f"Successfully converted and saved weights to {output_json_path}")
    except IOError as e:
        print(f"Error saving JSON to {output_json_path}. Error: {e}")
    except TypeError as e:
        print(f"Error serializing data to JSON (likely due to non-serializable types). Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch BitNet models to Rust-readable JSON format.")
    parser.add_argument("pytorch_model_path", type=str, help="Path to the PyTorch model checkpoint (.pth or .safetensors).")
    parser.add_argument("output_json_path", type=str, help="Path to save the converted JSON data.")
    # Add arguments for ModelArgs overrides if necessary
    parser.add_argument("--dim", type=int, help="Model dimension.")
    parser.add_argument("--n_layers", type=int, help="Number of layers.")
    parser.add_argument("--n_heads", type=int, help="Number of attention heads.")
    parser.add_argument("--vocab_size", type=int, help="Vocabulary size.")
    # ... other ModelArgs fields ...

    args = parser.parse_args()

    # Collect ModelArgs overrides from command line
    model_args_override: Dict[str, Any] = {}
    if args.dim is not None: model_args_override["dim"] = args.dim
    if args.n_layers is not None: model_args_override["n_layers"] = args.n_layers
    if args.n_heads is not None: model_args_override["n_heads"] = args.n_heads
    if args.vocab_size is not None: model_args_override["vocab_size"] = args.vocab_size
    # ...

    # --- Create a dummy PyTorch model state_dict for testing the script ---
    # This section should be removed when using a real model.
    # It's here to make the script runnable for demonstration without a real model file.
    def create_dummy_state_dict_and_args(override_args=None):
        print("Creating dummy state_dict for testing the conversion script...")
        dummy_args = {
            "dim": 64, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
            "vocab_size": 1000, "ffn_hidden_dim": None, "ffn_dim_multiplier": 2.0,
            "norm_eps": 1e-5, "rope_theta": 10000.0
        }
        if override_args:
            dummy_args.update(override_args)

        if dummy_args.get("ffn_hidden_dim") is None and dummy_args.get("ffn_dim_multiplier") is not None:
             dummy_args["ffn_hidden_dim"] = int(dummy_args["dim"] * dummy_args["ffn_dim_multiplier"])


        s = {}
        s["tok_embeddings.weight"] = torch.randn(dummy_args["vocab_size"], dummy_args["dim"])
        s["norm.weight"] = torch.randn(dummy_args["dim"])
        s["output.weight"] = torch.randn(dummy_args["vocab_size"], dummy_args["dim"]) # BitLinear

        for i in range(dummy_args["n_layers"]):
            # Attention
            s[f"layers.{i}.attention.wq.weight"] = torch.randn(dummy_args["n_heads"] * (dummy_args["dim"] // dummy_args["n_heads"]), dummy_args["dim"])
            s[f"layers.{i}.attention.wk.weight"] = torch.randn(dummy_args["n_kv_heads"] * (dummy_args["dim"] // dummy_args["n_heads"]), dummy_args["dim"])
            s[f"layers.{i}.attention.wv.weight"] = torch.randn(dummy_args["n_kv_heads"] * (dummy_args["dim"] // dummy_args["n_heads"]), dummy_args["dim"])
            s[f"layers.{i}.attention.wo.weight"] = torch.randn(dummy_args["dim"], dummy_args["n_heads"] * (dummy_args["dim"] // dummy_args["n_heads"]))
            s[f"layers.{i}.attention_norm.weight"] = torch.randn(dummy_args["dim"])
            # FeedForward (SwiGLU like: w1, w2, w3)
            # Hidden dim for FFN (e.g. SwiGLU style)
            ffn_h_dim = dummy_args["ffn_hidden_dim"]
            s[f"layers.{i}.feed_forward.w1.weight"] = torch.randn(ffn_h_dim, dummy_args["dim"])
            s[f"layers.{i}.feed_forward.w3.weight"] = torch.randn(ffn_h_dim, dummy_args["dim"]) # Gating
            s[f"layers.{i}.feed_forward.w2.weight"] = torch.randn(dummy_args["dim"], ffn_h_dim)
            s[f"layers.{i}.ffn_norm.weight"] = torch.randn(dummy_args["dim"])

        # Save this dummy state_dict to a file so the main function can load it
        dummy_path = "dummy_model.pth"
        torch.save(s, dummy_path)
        print(f"Dummy state_dict saved to {dummy_path}")
        return dummy_path, dummy_args

    # Replace args.pytorch_model_path with dummy if you want to test without a real model.
    # For real use, provide the actual path to your model.
    use_dummy_model = True
    if args.pytorch_model_path == "dummy": # Allow "dummy" as a special CLI arg for testing
        use_dummy_model = True
        print("Using dummy model for conversion test.")
    elif not Path(args.pytorch_model_path).exists(): # Basic check if not dummy
        use_dummy_model = True
        print(f"Warning: Path {args.pytorch_model_path} does not exist. Using dummy model for conversion test.")

    actual_model_path = args.pytorch_model_path
    if use_dummy_model:
        from pathlib import Path # For Path().exists()
        actual_model_path, dummy_model_args_for_override = create_dummy_state_dict_and_args(model_args_override)
        # Ensure model_args_override takes precedence if specific CLI args were given,
        # otherwise use the dummy_model_args.
        final_model_args_override = dummy_model_args_for_override.copy()
        final_model_args_override.update(model_args_override) # CLI args take precedence
        model_args_override = final_model_args_override


    # --- End of Dummy Model Section ---

    convert_model_to_rust_format(
        actual_model_path,
        args.output_json_path,
        model_args_override=model_args_override
    )

    # Clean up dummy model file if it was created
    if use_dummy_model and actual_model_path == "dummy_model.pth":
        try:
            Path(actual_model_path).unlink()
            print(f"Cleaned up dummy model file: {actual_model_path}")
        except OSError as e:
            print(f"Error cleaning up dummy model file: {e}")

```

**Erläuterungen und wichtige Punkte:**

1.  **`quantize_weights_ternary_python`:**
    *   Dies ist eine Python-Version der Quantisierungslogik. **Sie ist derzeit eine vereinfachte Version und spiegelt möglicherweise nicht exakt die 1.58-Bit-Quantisierung des BitNet-Papiers wider.** Die genaue Skalierung (z.B. `absmean` über den gesamten Tensor oder per Kanal) und die Rundungsmethode zu {-1, 0, 1} sind entscheidend. Ich habe eine Zeilen-basierte `absmean`-Skalierung und eine einfache Rundung als Platzhalter implementiert.
    *   Die Skalen werden als `absmax` der ursprünglichen `f32`-Gewichte pro Ausgabekanal berechnet. Dies ist eine gängige Methode.
    *   Das Packen der ternären Werte in 2-Bit-Darstellungen (00 für 0, 01 für 1, 10 für -1) erfolgt Bit für Bit. Die Reihenfolge (LSB zuerst) muss mit der Entpacklogik in Rust übereinstimmen.
2.  **`convert_model_to_rust_format`:**
    *   **Modell laden:** Versucht, `.safetensors` zu laden, und greift dann auf `.pth` zurück.
    *   **`ModelArgs`:** Die Modellkonfigurationsparameter (`dim`, `n_layers` usw.) müssen entweder aus der PyTorch-Checkpoint-Datei extrahiert (falls vorhanden) oder manuell übergeben werden. Ich habe einen `model_args_override`-Mechanismus und Standardwerte hinzugefügt. Die Berechnung von `ffn_hidden_dim` ist ebenfalls modellabhängig.
    *   **`key_map`:** Dies ist der **kritischste Teil** und **hochgradig modellspezifisch**. Es bildet die Namen der Layer/Parameter im PyTorch-Modell auf die Namen ab, die in der Rust-Struktur erwartet werden. Ich habe ein Beispiel-Mapping basierend auf typischen Llama-ähnlichen Architekturen eingefügt. Dies muss sorgfältig an das spezifische PyTorch-Modell angepasst werden.
    *   **Gewichtsverarbeitung:**
        *   Iteriert durch den `state_dict` des PyTorch-Modells.
        *   Bestimmt anhand des Rust-Schlüssels (und ob "weight" im Namen vorkommt), ob ein Gewicht zu einer `BitLinear`-Schicht gehört.
        *   `BitLinear`-Gewichte werden mit `quantize_weights_ternary_python` quantisiert und gepackt. Die gepackten Gewichte und ihre Skalen werden Base64-kodiert.
        *   Andere Gewichte (z.B. Embeddings, RMSNorm-Skalen) werden als `f32` belassen und ebenfalls Base64-kodiert.
        *   Forminformationen und Datentypen werden für die spätere Verifizierung und das Laden in Rust gespeichert.
    *   **Sanity Checks:** Ein einfacher Check wurde hinzugefügt, um zu prüfen, ob alle erwarteten Layer-Schlüssel basierend auf `n_layers` verarbeitet wurden.
    *   **Speichern:** Die konvertierten Daten (Modell-Argumente und Gewichte) werden in einer JSON-Datei gespeichert.
3.  **Dummy-Modell-Erstellung (`create_dummy_state_dict_and_args`, `if __name__ == "__main__":` Logik):**
    *   Um das Skript ohne ein echtes Modell testen zu können, habe ich Code hinzugefügt, der einen Dummy `state_dict` und Dummy-Argumente erstellt und in `dummy_model.pth` speichert.
    *   Das Skript kann mit `python convert_weights.py dummy dummy_output.json` ausgeführt werden, um diese Dummy-Logik zu verwenden.
    *   **Dieser Dummy-Teil sollte entfernt oder deaktiviert werden, wenn mit echten Modellen gearbeitet wird.**
4.  **Abhängigkeiten:** Dieses Skript benötigt `torch`, `numpy` und optional `safetensors`. Diese müssen in der Python-Umgebung installiert sein.

Dieses Skript ist ein erster Entwurf. Die `quantize_weights_ternary_python`-Funktion und insbesondere die `key_map` müssen sehr sorgfältig an das spezifische BitNet-PyTorch-Modell angepasst werden, das konvertiert werden soll. Die genaue 1.58-Bit-Quantisierungsstrategie aus dem Paper (Skalierung, Rundung) muss hier präzise implementiert werden, um kompatibel mit der Rust-Seite zu sein.
