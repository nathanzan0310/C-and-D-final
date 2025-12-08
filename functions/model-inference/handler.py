import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "/opt/model"
device = torch.device("cpu")

_tokenizer = None
_model = None

def _get_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model.to(device)
        model.eval()
        _tokenizer = tokenizer
        _model = model
    return _tokenizer, _model

def handle(event, context):
    try:
        raw = event.body or "{}"
        data = json.loads(raw)
        text = data.get("text", "")
        if not text.strip():
            return _resp(400, {"error": "missing non-empty 'text'"})

        tokenizer, model = _get_model()

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=int(data.get("max_length", 128)),
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        score, idx = torch.max(probs, dim=-1)

        id2label = model.config.id2label
        label = id2label.get(int(idx), str(int(idx)))

        return _resp(200, {
            "label": label,
            "score": float(score),
            "raw_logits": [float(x) for x in logits.tolist()],
        })
    except Exception as e:
        return _resp(500, {"error": str(e)})

def _resp(status, body):
    return {
        "statusCode": status,
        "body": json.dumps(body),
        "headers": {"Content-Type": "application/json"},
    }
