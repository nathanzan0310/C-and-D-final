from pathlib import Path
import os
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch

model_id = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# Set where Hugging Face downloads/caches model files.
cache_dir = Path(os.environ.get("HF_HOME", Path.cwd() / "hf-cache"))
cache_dir.mkdir(parents=True, exist_ok=True)

# Where to write the exported ONNX model. Override with ONNX_OUT env var.
output_dir = Path(os.environ.get("ONNX_OUT", Path.cwd()))
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "distilbert.onnx"

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_id, cache_dir=cache_dir)

dummy_input = tokenizer("hello world", return_tensors="pt")

torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    str(output_path),
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    opset_version=18,
    # Allow variable batch and sequence lengths at inference time.
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch"},
    },
)
