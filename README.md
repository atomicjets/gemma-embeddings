# High-Throughput Tweet Embedding Pipeline

This document outlines a high-performance, file-based architecture for generating embeddings for a large corpus of tweets, designed to maximize GPU utilization by bypassing database bottlenecks.

## 1. Architecture Overview

The system is designed for maximum throughput by decoupling data extraction from processing.

1.  **Export Phase:** Data is first exported from its source (e.g., MongoDB or BigQuery) into a line-delimited JSON file. This file contains only the `_id` and `text` for each document to be processed. This step is performed once.

2.  **Processing Phase:** The `embed_tweets.py` script reads from the JSON file, sends the text to a load-balanced pool of TEI servers, and writes the resulting embeddings to a series of Parquet files. This phase is fully resumable.

3.  **Consolidation Phase:** After processing is complete, the `consolidate_embeddings.py` script merges the many small Parquet files into a smaller number of large, multi-gigabyte files suitable for ingestion into data warehouses like BigQuery.

### Storage Layout

- **Input Data (NVMe):** The exported JSON file (e.g., `/mnt/raid1/exports/tweets_to_process_uk.json`) resides on fast storage to ensure the producer can read data quickly.
- **Temporary Files (NVMe):** `/mnt/raid1/embeddings/tmp` is used for staging Parquet files before they are moved.
- **Progress Markers (NVMe):** `/mnt/raid1/embeddings/progress` stores small `.done` files to track the progress of resumable chunks.
- **Final Output (SSD):** `/mnt/sdb_mount_point/embeddings/shards` is the destination for the final Parquet files.

## 2. Workflow

### Step 1: Export Data

First, export the data to be processed into a line-delimited JSON file. Each line should be a JSON object containing at least an `_id` and `text` field.

### Step 2: Run the Embedding Script

Launch the main processing script, pointing it to your input file and your load-balanced TEI endpoint.

```bash
python embed_tweets.py \
  --input-file /mnt/raid1/exports/tweets_to_process_uk.json \
  --tei-urls "http://127.0.0.1:18090/embed"
```

The script is resumable. If it is stopped, you can simply run the same command again, and it will pick up from the last completed chunk.

### Step 3: Consolidate the Output

Once the embedding script is finished, run the consolidation script to merge the output files.

```bash
python consolidate_embeddings.py
```

## 3. Real-World Performance

By bypassing the database bottleneck, this architecture allows the GPUs to be fully saturated. The system achieves a stable throughput of approximately **2,400 - 2,900 docs/second** on a 4x RTX 4090 setup.

```
2025-10-28 16:36:09,731 - INFO - Inst. speed: 2927.19 docs/s | Avg speed: 1758.32 docs/s | Total processed: 19104
2025-10-28 16:36:14,751 - INFO - Inst. speed: 2875.38 docs/s | Avg speed: 2111.29 docs/s | Total processed: 33536
2025-10-28 16:36:19,752 - INFO - Inst. speed: 2489.01 docs/s | Avg speed: 2201.74 docs/s | Total processed: 45984
2025-10-28 16:36:24,766 - INFO - Inst. speed: 2578.26 docs/s | Avg speed: 2274.64 docs/s | Total processed: 58912
2025-10-28 16:36:29,766 - INFO - Inst. speed: 2943.71 docs/s | Avg speed: 2382.91 docs/s | Total processed: 73632
