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
