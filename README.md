# Tweet Embedding Pipeline

This document outlines the architecture and process for generating embeddings for a large corpus of tweets using a multi-GPU setup and a two-tier storage system.

## 1. Architecture Overview

The system is designed for high-throughput, resumable embedding generation.

- **Hot Storage (NVMe):** `/mnt/raid1` is used for temporary files, model caches, and job queue metadata. This provides the low-latency I/O needed for these operations.
- **Cold Storage (SSD):** `/mnt/sdb_mount_point` is used for the final, large Parquet files containing the embeddings. This provides fast sequential write performance for the final output.

## 2. Database Setup (MongoDB)

Efficient processing and monitoring are handled by two key indexes in the `twitter_data` database. These should be created on each collection you intend to process (e.g., `tweets` and `tweets_usa`).

### Work Queue Index (Partial)

This is the primary index used by the script to find unprocessed documents. It's a partial index, which means it's very small and efficient as it only contains documents that meet the filter criteria.

```javascript
// In mongosh, connected to the 'twitter_data' db
db.tweets.createIndex(
  { lang: 1, classified: 1, _id: 1 },
  {
    name: 'embedding_work_queue',
    partialFilterExpression: {
      retweeted: null,
      gemma_embedded_768: null
    }
  }
)
```

### Monitoring Index (Sparse)

This index is used to efficiently count the number of documents that have already been processed. It's a sparse index, so it only includes documents that have the `gemma_embedded_768` field.

```javascript
// In mongosh, connected to the 'twitter_data' db
db.tweets.createIndex(
  { gemma_embedded_768: 1 },
  {
    sparse: true,
    name: "processed_monitor_idx"
  }
)
```

## 3. Running the Embedding Script

The `embed_tweets.py` script orchestrates the entire process. It should be run with the optimal parameters discovered during the auto-tuning phase.

```bash
# Example for the 'tweets' collection
python embed_tweets.py \
  --collection tweets \
  --tei-urls "http://localhost:18080/embed,http://localhost:18081/embed,http://localhost:18082/embed,http://localhost:18083/embed" \
  --batch-size 32
```

## 4. Consolidating the Output Files

After the embedding process is complete, the output directory will contain millions of small Parquet files. The `consolidate_embeddings.py` script is used to merge these into a smaller number of large, manageable files.

```bash
# Consolidate the files
python consolidate_embeddings.py
```

This will read from the `shards` directory and write new, larger files to the `consolidated` directory.

## 5. Real-World Performance

The following logs were captured from a live run of the high-performance, concurrent `embed_tweets.py` script. The system achieved a stable throughput of approximately **2,600 docs/second**.

```
2025-10-27 16:21:27,172 - INFO - Inst. speed: 2132.59 docs/s | Avg speed: 2132.59 docs/s | Total processed: 12832
2025-10-27 16:21:32,175 - INFO - Inst. speed: 3076.42 docs/s | Avg speed: 2561.09 docs/s | Total processed: 28224
2025-10-27 16:21:37,176 - INFO - Inst. speed: 2636.27 docs/s | Avg speed: 2584.56 docs/s | Total processed: 41408
2025-10-27 16:21:42,200 - INFO - Inst. speed: 2656.37 docs/s | Avg speed: 2601.70 docs/s | Total processed: 54752
2025-10-27 16:21:47,264 - INFO - Inst. speed: 2799.24 docs/s | Avg speed: 2640.02 docs/s | Total processed: 68928
2025-10-27 16:21:52,264 - INFO - Inst. speed: 2617.48 docs/s | Avg speed: 2636.39 docs/s | Total processed: 82016
2025-10-27 16:21:57,264 - INFO - Inst. speed: 2803.13 docs/s | Avg speed: 2659.48 docs/s | Total processed: 96032
2025-10-27 16:22:02,285 - INFO - Inst. speed: 2575.14 docs/s | Avg speed: 2649.19 docs/s | Total processed: 108960
2025-10-27 16:22:07,314 - INFO - Inst. speed: 2704.05 docs/s | Avg speed: 2655.17 docs/s | Total processed: 122560
2025-10-27 16:22:12,342 - INFO - Inst. speed: 2463.24 docs/s | Avg speed: 2636.32 docs/s | Total processed: 134944
2025-10-27 16:22:17,342 - INFO - Inst. speed: 2489.34 docs/s | Avg speed: 2623.23 docs/s | Total processed: 147392
2025-10-27 16:22:22,417 - INFO - Inst. speed: 2724.28 docs/s | Avg speed: 2631.60 docs/s | Total processed: 161216
2025-10-27 16:22:27,454 - INFO - Inst. speed: 2686.98 docs/s | Avg speed: 2635.81 docs/s | Total processed: 174752
2025-10-27 16:22:32,492 - INFO - Inst. speed: 2763.19 docs/s | Avg speed: 2644.81 docs/s | Total processed: 188672
2025-10-27 16:22:37,494 - INFO - Inst. speed: 2674.15 docs/s | Avg speed: 2646.73 docs/s | Total processed: 202048
2025-10-27 16:22:42,495 - INFO - Inst. speed: 2553.09 docs/s | Avg speed: 2640.97 docs/s | Total processed: 214816
2025-10-27 16:22:47,499 - INFO - Inst. speed: 2455.72 docs/s | Avg speed: 2630.24 docs/s | Total processed: 227104
2025-10-27 16:22:52,625 - INFO - Inst. speed: 2502.92 docs/s | Avg speed: 2623.10 docs/s | Total processed: 239936
2025-10-27 16:22:57,677 - INFO - Inst. speed: 2901.26 docs/s | Avg speed: 2637.66 docs/s | Total processed: 254592
2025-10-27 16:23:02,683 - INFO - Inst. speed: 2294.77 docs/s | Avg speed: 2620.75 docs/s | Total processed: 266080
2025-10-27 16:23:07,909 - INFO - Inst. speed: 2179.80 docs/s | Avg speed: 2599.16 docs/s | Total processed: 277472
2025-10-27 16:23:12,919 - INFO - Inst. speed: 2932.16 docs/s | Avg speed: 2614.09 docs/s | Total processed: 292160
2025-10-27 16:23:17,978 - INFO - Inst. speed: 2182.01 docs/s | Avg speed: 2595.38 docs/s | Total processed: 303200
2025-10-27 16:23:23,025 - INFO - Inst. speed: 2903.77 docs/s | Avg speed: 2608.15 docs/s | Total processed: 317856
2025-10-27 16:23:28,033 - INFO - Inst. speed: 2658.50 docs/s | Avg speed: 2610.14 docs/s | Total processed: 331168
2025-10-27 16:23:33,034 - INFO - Inst. speed: 2143.61 docs/s | Avg speed: 2592.44 docs/s | Total processed: 341888
2025-10-27 16:23:38,037 - INFO - Inst. speed: 2437.00 docs/s | Avg speed: 2586.76 docs/s | Total processed: 354080
2025-10-27 16:23:43,116 - INFO - Inst. speed: 2513.72 docs/s | Avg speed: 2584.15 docs/s | Total processed: 366848
2025-10-27 16:23:48,127 - INFO - Inst. speed: 2688.47 docs/s | Avg speed: 2587.71 docs/s | Total processed: 380320
2025-10-27 16:23:53,127 - INFO - Inst. speed: 2559.86 docs/s | Avg speed: 2586.79 docs/s | Total processed: 393120
2025-10-27 16:23:58,150 - INFO - Inst. speed: 2204.39 docs/s | Avg speed: 2574.56 docs/s | Total processed: 404192
2025-10-27 16:24:03,166 - INFO - Inst. speed: 2500.89 docs/s | Avg speed: 2572.28 docs/s | Total processed: 416736
```

---

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
