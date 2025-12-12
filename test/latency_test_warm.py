import argparse
import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from transformers import AutoTokenizer


def _load_trace(trace_path: Path, scale_time: float, limit: Optional[int]):
    """Load trace and return DataFrame with seconds-from-start column."""
    if trace_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(trace_path)
    else:
        df = pd.read_csv(trace_path)

    if "TIMESTAMP" not in df.columns:
        raise ValueError("trace is missing TIMESTAMP column")
    if "token_bucket" not in df.columns and "ContextTokens" not in df.columns:
        raise ValueError("trace is missing token_bucket or ContextTokens column")

    if "token_bucket" not in df.columns:
        # Bucketize on the fly if only ContextTokens is present
        bins = [0, 256, 1000, 4000, 8000]
        labels = ["small", "medium", "large", "xl"]
        df["token_bucket"] = pd.cut(df["ContextTokens"], bins=bins, labels=labels, include_lowest=True)

    ts = pd.to_datetime(df["TIMESTAMP"], utc=True, format="ISO8601", errors="coerce")
    if ts.isna().any():
        raise ValueError("trace TIMESTAMP contains unparsable rows; consider cleaning input")
    t0 = ts.iloc[0]
    df["ts_seconds"] = (ts - t0).dt.total_seconds() * scale_time

    if limit is not None:
        df = df.iloc[:limit]
    return df


def replay(trace_path: Path, prompts_path: Path, invoke_url: str, output_csv: Path, timeout: float, scale_time: float, limit: Optional[int]):
    prompts = json.loads(Path(prompts_path).read_text())
    df = _load_trace(trace_path, scale_time, limit)

    cache_dir = (Path(__file__).resolve().parent.parent / "hf-cache").as_posix()
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english",
        cache_dir=cache_dir,
    )

    if not invoke_url.endswith("/infer_warm"):
        invoke_url = invoke_url.rstrip("/") + "/infer_warm"

    start = time.time()
    results = []
    for _, row in df.iterrows():
        delay = (start + row["ts_seconds"]) - time.time()
        if delay > 0:
            time.sleep(delay)

        bucket = row["token_bucket"]
        prompt_text = prompts.get(str(bucket), prompts.get("small", ""))
        encoded = tokenizer(prompt_text, return_tensors="np", truncation=True)
        input_ids = encoded["input_ids"][0].tolist()
        attention_mask = encoded["attention_mask"][0].tolist()

        payload = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bucket": str(bucket),
            "timestamp": row["TIMESTAMP"],
        }

        t0 = time.time()
        status = "error"
        latency = None
        error_msg = None
        try:
            resp = requests.post(invoke_url, json=payload, timeout=timeout)
            status = resp.status_code
            latency = time.time() - t0
        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc)

        results.append({
            "ts_seconds": row["ts_seconds"],
            "bucket": str(bucket),
            "status": status,
            "latency_s": latency,
            "error": error_msg,
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Wrote {output_csv} with {len(results)} rows")


def main():
    parser = argparse.ArgumentParser(description="Replay trace against /infer_warm and record latency.")
    parser.add_argument("--trace", default="invocation/AzureTrace.parquet", help="Path to trace parquet/csv")
    parser.add_argument("--prompts", default="invocation/prompts.json", help="Path to prompts mapping")
    parser.add_argument("--url", required=True, help="Base URL of the gateway (will append /infer_warm if missing)")
    parser.add_argument("--out", default="invocation/replay_latencies_warm.csv", help="Where to write results CSV")
    parser.add_argument("--timeout", type=float, default=10.0, help="Per-request timeout seconds")
    parser.add_argument("--scale-time", type=float, default=1.0, help="Scale factor for inter-arrival times")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of requests")
    args = parser.parse_args()

    replay(
        trace_path=Path(args.trace),
        prompts_path=Path(args.prompts),
        invoke_url=args.url,
        output_csv=Path(args.out),
        timeout=args.timeout,
        scale_time=args.scale_time,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
