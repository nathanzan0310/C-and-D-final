import time
import requests

def replay_trace(data_stream, invoke_url, timeout=1.0):
    """
    data_stream: iterable of dicts like:
      {"timestamp": <seconds>, "tokens": <int>}

    invoke_url: full HTTP endpoint of the serverless function
    """

    start_time = time.time()

    for row in data_stream:
        ts = float(row["TIMESTAMP"])
        tokens = row["ContextTokens"]

        # Pace execution according to trace timestamp
        now = time.time()
        delay = (start_time + ts) - now
        if delay > 0:
            time.sleep(delay)

        payload = {
            "timestamp": ts,
            "tokens": tokens
        }

        try:
            # invocation TODO: Figure out format
            requests.post(invoke_url, json=payload, timeout=timeout)
        except Exception:
            # Ignore network errors to keep replay going
            pass
