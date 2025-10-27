# GPU Embedding Benchmark

## Server Performance Results (4x 4090 GPUs)

The following are the results from the `autotune_embed.py` script on a server with 4x 4090 GPUs.

| Batch Size | Concurrency | Docs/s | Tok/s  | p50 (s) | p95 (s) | Errors | Total Time (s) |
|------------|-------------|--------|--------|---------|---------|--------|----------------|
| 32         | 128         | 174    | 72,711 | 21.91   | 25.29   | 0      | 88.39          |
| 32         | 192         | 186    | 76,873 | 31.69   | 35.71   | 0      | 107.70         |
| 32         | 256         | 178    | 73,633 | 42.94   | 47.27   | 0      | 112.20         |
| 32         | 384         | 188    | 78,018 | 59.13   | 60.85   | 201    | 106.36         |

**Note:** No combination met the p95 ≤ 10.0s requirement without errors. It is recommended to consider lowering the `BATCH` size or the server's `--max-batch-tokens` setting.
