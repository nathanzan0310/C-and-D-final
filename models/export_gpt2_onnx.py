from pathlib import Path
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
model_id = "gpt2"
output_filename = "gpt2.onnx"

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
cache_dir = Path(os.environ.get("HF_HOME", Path.cwd() / "hf-cache"))
output_dir = Path(os.environ.get("ONNX_OUT", Path.cwd()))
cache_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / output_filename

print(f"Exporting {model_id} → {output_path}")

# ---------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    torch_dtype=torch.float32,
)

model.eval()
model.config.use_cache = True

# ---------------------------------------------------------------------
# Wrapper (CRITICAL)
# ---------------------------------------------------------------------
class GPT2ONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.num_layers = model.config.n_layer

    def forward(self, input_ids, attention_mask, *past):
        past_key_values = tuple(
            (past[2 * i], past[2 * i + 1])
            for i in range(self.num_layers)
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        flat_present = tuple(
            t for layer in outputs.past_key_values for t in layer
        )

        return (outputs.logits,) + flat_present


wrapped_model = GPT2ONNXWrapper(model)
wrapped_model.eval()

# ---------------------------------------------------------------------
# Dummy inputs (single-token decode)
# ---------------------------------------------------------------------
num_layers = model.config.n_layer
num_heads = model.config.n_head
head_dim = model.config.n_embd // num_heads
past_seq_len = 8  # dummy cache length

input_ids = torch.ones((1, 1), dtype=torch.long)
attention_mask = torch.ones((1, 1), dtype=torch.long)

past_shape = (1, num_heads, past_seq_len, head_dim)

past_key_values = tuple(
    (torch.zeros(past_shape), torch.zeros(past_shape))
    for _ in range(num_layers)
)

flattened_past = tuple(t for layer in past_key_values for t in layer)
model_args = (input_ids, attention_mask) + flattened_past

# ---------------------------------------------------------------------
# ONNX IO names + dynamic axes
# ---------------------------------------------------------------------
input_names = ["input_ids", "attention_mask"]
output_names = ["logits"]

dynamic_axes = {
    "input_ids": {0: "batch"},
    "attention_mask": {0: "batch"},
    "logits": {0: "batch", 1: "sequence"},
}

for i in range(num_layers):
    input_names += [f"past.{i}.key", f"past.{i}.value"]
    output_names += [f"present.{i}.key", f"present.{i}.value"]

    dynamic_axes[f"past.{i}.key"] = {0: "batch", 2: "past_sequence"}
    dynamic_axes[f"past.{i}.value"] = {0: "batch", 2: "past_sequence"}
    dynamic_axes[f"present.{i}.key"] = {0: "batch", 2: "past_sequence"}
    dynamic_axes[f"present.{i}.value"] = {0: "batch", 2: "past_sequence"}

# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------
torch.onnx.export(
    wrapped_model,
    model_args,
    output_path.as_posix(),
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    opset_version=13,
    export_params=True,
)

print("✅ GPT-2 ONNX export complete")
