from pathlib import Path
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------
# Configuration
# --------------------------------------------------
model_id = "distilbert/distilgpt2"
output_filename = "distilgpt2.onnx"

# --------------------------------------------------
# Setup
# --------------------------------------------------
cache_dir = Path(os.environ.get("HF_HOME", Path.cwd() / "hf-cache"))
output_dir = Path(os.environ.get("ONNX_OUT", Path.cwd()))
cache_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / output_filename

print(f"Starting export for {model_id} (Generation with Cache) to {output_path}")

# --------------------------------------------------
# Load model
# --------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir)
model.config.use_cache = True
model.eval()

# --------------------------------------------------
# TorchScript-safe wrapper (EXPLICIT CACHE ARGS)
# distilgpt2 has n_layer = 6
# --------------------------------------------------
class DistilGPT2WithPast(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids,
        attention_mask,
        past_0_key, past_0_value,
        past_1_key, past_1_value,
        past_2_key, past_2_value,
        past_3_key, past_3_value,
        past_4_key, past_4_value,
        past_5_key, past_5_value,
    ):
        past_key_values = (
            (past_0_key, past_0_value),
            (past_1_key, past_1_value),
            (past_2_key, past_2_value),
            (past_3_key, past_3_value),
            (past_4_key, past_4_value),
            (past_5_key, past_5_value),
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        # Return logits + flattened present cache
        return (
            outputs.logits,
            outputs.past_key_values[0][0], outputs.past_key_values[0][1],
            outputs.past_key_values[1][0], outputs.past_key_values[1][1],
            outputs.past_key_values[2][0], outputs.past_key_values[2][1],
            outputs.past_key_values[3][0], outputs.past_key_values[3][1],
            outputs.past_key_values[4][0], outputs.past_key_values[4][1],
            outputs.past_key_values[5][0], outputs.past_key_values[5][1],
        )


wrapped_model = DistilGPT2WithPast(model)
wrapped_model.eval()

# --------------------------------------------------
# Dummy inputs (single-token decode)
# --------------------------------------------------
num_heads = model.config.n_head
head_dim = model.config.n_embd // num_heads
past_seq_len = 10

input_ids = torch.ones((1, 1), dtype=torch.long)
attention_mask = torch.ones((1, 1), dtype=torch.long)

past_shape = (1, num_heads, past_seq_len, head_dim)

past = [torch.randn(past_shape) for _ in range(12)]

model_args = (
    input_ids,
    attention_mask,
    *past,
)

# --------------------------------------------------
# Names and dynamic axes
# --------------------------------------------------
input_names = ["input_ids", "attention_mask"]
output_names = ["logits"]

dynamic_axes = {
    "input_ids": {0: "batch"},
    "attention_mask": {0: "batch"},
    "logits": {0: "batch", 1: "sequence"},
}

for i in range(6):
    input_names += [f"past.{i}.key", f"past.{i}.value"]
    output_names += [f"present.{i}.key", f"present.{i}.value"]

    dynamic_axes[f"past.{i}.key"] = {0: "batch", 2: "past_sequence"}
    dynamic_axes[f"past.{i}.value"] = {0: "batch", 2: "past_sequence"}
    dynamic_axes[f"present.{i}.key"] = {0: "batch", 2: "past_sequence"}
    dynamic_axes[f"present.{i}.value"] = {0: "batch", 2: "past_sequence"}

# --------------------------------------------------
# Export to ONNX (LEGACY EXPORTER)
# --------------------------------------------------
from torch.onnx import utils as onnx_utils

onnx_utils.export(
    wrapped_model,
    model_args,
    str(output_path),
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    opset_version=13,
)

print(f"✅ Exported DistilGPT2 to {output_path}")
