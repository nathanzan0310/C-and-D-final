import time
import argparse
import json
import csv

import requests
import pandas as pd
from tqdm import tqdm


def build_buckets(df):
    """
    According to ContextTokens, devide into 4 buckets:
      1: min 25%
      2: 25% ~ 50%
      3: 50% ~ 75%
      4: max 25%
    return (quantiles, bucket_fn)
    """
    q = df["ContextTokens"].quantile([0.25, 0.5, 0.75])
    q1, q2, q3 = q[0.25], q[0.5], q[0.75]

    def bucket_for(tokens: int) -> str:
        if tokens <= q1:
            return "1"
        elif tokens <= q2:
            return "2"
        elif tokens <= q3:
            return "3"
        else:
            return "4"

    return (q1, q2, q3), bucket_for


def load_trace(parquet_path, prompts_path, limit=None):
    """
    Read the files AzureTrace.parquet and prompts.json, and return the list of events.
    For each event: 
    {
        "timestamp": <Relative time in seconds> 
        "bucket_id": "1".." 4" 
        "prompt": <string>,
        "context_tokens": int,
        "generated_tokens": int,
    }
    """
    print(f"[INFO] Loading trace from {parquet_path} ...")
    df = pd.read_parquet(parquet_path)

    # Sort by time
    df = df.sort_values("TIMESTAMP")

    # Compute bucket threshold
    (q1, q2, q3), bucket_for = build_buckets(df)
    print(f"[INFO] Bucket thresholds (ContextTokens):")
    print(f"       bucket 1: <= {q1:.1f}")
    print(f"       bucket 2: <= {q2:.1f}")
    print(f"       bucket 3: <= {q3:.1f}")
    print(f"       bucket 4:  > {q3:.1f}")

    if limit is not None:
        df = df.head(limit)

    # Relative time
    ts0 = pd.to_datetime(df["TIMESTAMP"].iloc[0])
    ts_all = pd.to_datetime(df["TIMESTAMP"])
    df["rel_ts"] = (ts_all - ts0).dt.total_seconds()

    # Read prompts.json
    with open(prompts_path, "r") as f:
        prompts = json.load(f)

    events = []
    for _, row in df.iterrows():
        context_tokens = int(row["ContextTokens"])
        generated_tokens = int(row["GeneratedTokens"])

        bucket_id = bucket_for(context_tokens)  # "1".."4"
        if bucket_id not in prompts:
            raise ValueError(f"Bucket id {bucket_id} not found in prompts.json keys {list(prompts.keys())}")

        prompt_text = prompts[bucket_id]

        events.append(
            {
                "timestamp": float(row["rel_ts"]),
                "bucket_id": bucket_id,
                "prompt": prompt_text,
                "context_tokens": context_tokens,
                "generated_tokens": generated_tokens,
            }
        )
    return events


def replay_trace(
    events,
    invoke_url,
    auth_tuple=None,
    timeout=1.0,
    scale=1.0,
    blocking=False,
    output_csv=None,
):
    """
    events: list of dicts as returned by load_trace
    invoke_url: 
      https://localhost:31001/api/v1/namespaces/_/actions/hello?blocking=false
    """
    t_start = time.time()
    start_time = time.time()
    rows_for_csv = []

    if blocking:
        invoke_url_blocking = invoke_url.replace("blocking=false", "blocking=true")
    else:
        invoke_url_blocking = invoke_url

    for ev in tqdm(events, desc="Replaying trace"):
        ts = ev["timestamp"] * scale
        bucket_id = ev["bucket_id"]
        prompt_text = ev["prompt"]
        context_tokens = ev["context_tokens"]
        generated_tokens = ev["generated_tokens"]

        now = time.time()
        target = start_time + ts
        delay = target - now
        if delay > 0:
            time.sleep(delay)

        payload = {
            "timestamp": ts,
            "bucket_id": bucket_id,
            "prompt": prompt_text,
            "context_tokens": context_tokens,
            "generated_tokens": generated_tokens,
        }

        try:
            if blocking:
                t0 = time.time()
                resp = requests.post(
                    invoke_url_blocking,
                    json=payload,
                    timeout=timeout,
                    auth=auth_tuple,
                    verify=False,
                )
                t1 = time.time()
                client_latency = t1 - t0

                ow_duration = None
                try:
                    data = resp.json()
                    # REST API
                    ow_duration = data.get("duration", None)
                except Exception:
                    pass

                rows_for_csv.append(
                    {
                        "rel_timestamp": ts,
                        "bucket_id": bucket_id,
                        "context_tokens": context_tokens,
                        "generated_tokens": generated_tokens,
                        "client_latency_sec": client_latency,
                        "ow_duration_ms": ow_duration,
                        "status_code": resp.status_code,
                    }
                )
            else:
                # fire-and-forget
                requests.post(
                    invoke_url,
                    json=payload,
                    timeout=timeout,
                    auth=auth_tuple,
                    verify=False,
                )
        except Exception as e:
            print(f"[WARN] request failed: {e}")

    if blocking and output_csv is not None and rows_for_csv:
        fieldnames = list(rows_for_csv[0].keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_for_csv)
        print(f"[INFO] Wrote per-request stats to {output_csv}")
        
    t_end = time.time()
    print(f"[INFO] Workload end-to-end time: {t_end - t_start:.3f} seconds")



def main():
    parser = argparse.ArgumentParser(
        description="Replay Azure inference trace against an OpenWhisk action"
    )
    parser.add_argument(
        "--trace", required=True, help="Path to AzureTrace.parquet"
    )
    parser.add_argument(
        "--prompts",
        required=True,
        help="Path to prompts.json (bucket_id -> prompt text)",
    )
    parser.add_argument(
        "--api-host",
        required=True,
        help="OpenWhisk API host, e.g. https://localhost:31001",
    )
    parser.add_argument(
        "--auth",
        required=True,
        help="OpenWhisk auth key, format: uuid:key",
    )
    parser.add_argument(
        "--namespace",
        default="_",
        help="OpenWhisk namespace (default: _ / guest)",
    )
    parser.add_argument(
        "--action",
        required=True,
        help="Action name to invoke (e.g. hello or model-serving action)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="HTTP request timeout in seconds",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Time scaling factor (e.g., 0.1 = 10x faster than trace)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Limit number of events for testing (default: 100)",
    )
    parser.add_argument(
        "--blocking",
        action="store_true",
        help="Use blocking=true and measure per-request latency",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="If set (and --blocking), write per-request stats to this CSV file",
    )
    args = parser.parse_args()

    # Parse auth: "uuid:key"
    if ":" not in args.auth:
        raise ValueError("Auth must be in format 'uuid:key'")
    user, key = args.auth.split(":", 1)
    auth_tuple = (user, key)

    invoke_url = (
        f"{args.api_host}/api/v1/namespaces/{args.namespace}/actions/"
        f"{args.action}?blocking=false"
    )

    events = load_trace(args.trace, args.prompts, limit=args.limit)
    print(f"[INFO] Loaded {len(events)} events (after limit).")
    print(f"[INFO] Replaying against {invoke_url}")
    print(f"[INFO] Using time scale factor = {args.scale}")
    print(f"[INFO] Blocking mode: {args.blocking}")

    replay_trace(
        events,
        invoke_url=invoke_url,
        auth_tuple=auth_tuple,
        timeout=args.timeout,
        scale=args.scale,
        blocking=args.blocking,
        output_csv=args.output_csv,
    )

    print("[INFO] Replay finished.")


if __name__ == "__main__":
    main()
