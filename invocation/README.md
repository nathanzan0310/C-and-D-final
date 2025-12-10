AzureTrace.parquet: inference data. (head of dataset below)
| TIMESTAMP                       | token_bucket |
|---------------------------------|-------------|
| 2024-05-12 00:00:00.001163+00:00 | large       |
| 2024-05-12 00:00:00.041683+00:00 | medium      |
| 2024-05-12 00:00:00.157988+00:00 | medium      |
| 2024-05-12 00:00:00.158932+00:00 | large       |
| 2024-05-12 00:00:00.248279+00:00 | medium      |


prompts.json: mapping from token_bucket size to corresponding prompt to invoke.
e.g. {"small": [prompt here], "medium": [ slightly longer prompt here], ...}

invokepattern.py : INCOMPLETE
convert.py: Processing (NOT NEEDED IF AzureTrace.parquet IS THERE)
