Nice—let’s spin Gemma back up on your 4090 and kick the autotuner.

## 0) One-time checklist

* On Hugging Face, **accept the model terms** for `google/embeddinggemma-300m`.
* Grab a **HF read token**.

## 1) Start TEI (Gemma-300M on a 4090)

```bash
# clean old container (safe if none exists)
docker rm -f tei-gemma 2>/dev/null || true

# run Gemma-300M with Candle backend (fp32 only), tuned for 4090
docker run --name tei-gemma --gpus all -p 8080:80 \
  -e HF_TOKEN=... \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -e TOKENIZATION_WORKERS=$(nproc) \
  -v /srv/tei-cache:/data \
  ghcr.io/huggingface/text-embeddings-inference:89-1.8 \
    --model-id google/embeddinggemma-300m \
    --dtype float32 \
    --pooling mean \
    --max-batch-tokens 32768 \
    --max-client-batch-size 1024 \
    --max-concurrent-requests 512 \
    --auto-truncate
```

### Health & smoke test

```bash
docker logs -n 60 tei-gemma
curl http://localhost:8080/health         # -> ok
curl -s -X POST http://localhost:8080/embed \
  -H "Content-Type: application/json" \
  -d '{"inputs":["hello","bonjour"]}' | jq '.embeddings | length'
```

> Notes
> • Gemma on Candle is **fp32 only** (so `--dtype float32`).
> • `--normalize` is not a valid flag in 1.8; mean pooling is set via `--pooling mean`.
> • If you have headroom, you can bump `--max-batch-tokens` to **65536** later.

## 2) Run your autotuner

From your repo/folder with `autotune_embed.py`:

```bash
export TEI_URL=http://localhost:8080/embed
export MODEL_ID=google/embeddinggemma-300m

# Throughput mode (good starting grid)
export BATCH_GRID=16,24,32
export CONC_GRID=64,96,128,160
export P95_TARGET=10
export TRIAL_SECS=20
export TRIAL_DOCS=20000

python autotune_embed.py
```

### If you want lower latency trials

Reduce per-request work and increase inflight:

```bash
# server: smaller per-request tokens
docker update tei-gemma >/dev/null 2>&1 || true
docker rm -f tei-gemma && docker run -d --name tei-gemma --gpus all -p 8080:80 \
  -e HF_TOKEN=hf_xxxxxxxx -e HF_HUB_ENABLE_HF_TRANSFER=1 -e TOKENIZATION_WORKERS=$(nproc) \
  -v /srv/tei-cache:/data \
  ghcr.io/huggingface/text-embeddings-inference:89-1.8 \
    --model-id google/embeddinggemma-300m --dtype float32 --pooling mean \
    --max-batch-tokens 16384 --max-client-batch-size 2048 --max-concurrent-requests 512 --auto-truncate

# client: many small requests
export BATCH_GRID=8,12,16,24
export CONC_GRID=96,128,160,192
export P95_TARGET=6
python autotune_embed.py
```

## 3) What “good” looks like on a single 4090 (fp32 Gemma)

* **Throughput mode:** p50 ≈ 5–12 s, p95 ≤ 12–15 s, **~80–120k tokens/s**.
* **Latency mode:** p50 ≈ 1–4 s, p95 ≤ 6–8 s, **~50–90k tokens/s**.
* Watch `nvidia-smi`; aim for **95–100% GPU util**. If util is low, raise `CONC`; if p95 balloons, lower `BATCH` or `--max-batch-tokens`.

## 4) After you pick a winner

Grab the best pair from the autotuner (highest tokens/sec under your p95 target) and lock them in your producer:

```bash
# example outcome, adjust to your winner
export BATCH=24
export CONC=96
python run.py
```

## 5) (Optional) Ready for the INT8 step later

Once bulk embedding is done and you’ve ingested to Qdrant/BigQuery, you can quantise (in Qdrant or offline) to hit your memory target. That doesn’t affect TEI—only storage and search.

---

If anything errors (401, 404, dtype, normalise flag), it’s usually one of: wrong model id (`google/embeddinggemma-300m`), licence not accepted, or using `bfloat16/float16` on Gemma (not supported).



Here’s a concise summary of your **best-known settings for 1 × RTX 4090** (EmbeddingGemma-300M, TEI v1.8 fp32 build):

---

### 🏆 Best throughput (stable)

| Parameter                       | Value                                | Notes                                                                      |
| ------------------------------- | ------------------------------------ | -------------------------------------------------------------------------- |
| **Batch size**                  | **32**                               | Larger batches give fuller GPU utilisation without hitting errors.         |
| **Concurrency (CONC)**          | **96**                               | Balances throughput and latency; higher values start to cause instability. |
| **Throughput**                  | **≈ 512 docs/s  (≈ 209 k tokens/s)** | Best tokens/s achieved with zero errors.                                   |
| **Latency (p50 / p95)**         | **49.0 s / 55.2 s**                  | Long end-to-end, but fine for offline embedding.                           |
| **Errors**                      | 0                                    | Clean run.                                                                 |
| **Scaling efficiency baseline** | 1.00×                                | Use this as your 1-GPU reference.                                          |

---

### ⚙️ Good fallback (lower latency, still high throughput)

| Parameter                     | Value                                        | Metrics                                                  |
| ----------------------------- | -------------------------------------------- | -------------------------------------------------------- |
| **Batch = 24**, **Conc = 96** | ≈ 390 docs/s  (≈ 166 k tokens/s), p95 ≈ 43 s | Smooth and predictable; use if you want tighter latency. |

---

### 🧠 Derived performance estimates

| Target corpus    | Assumed avg tokens/tweet        | Approx runtime |
| ---------------- | ------------------------------- | -------------- |
| **400 M tweets** | 15 tok (avg tweet)              | ~ 8 h          |
|                  | 20 tok                          | ~ 10.6 h       |
|                  | 25 tok                          | ~ 13.3 h       |
|                  | 30 tok                          | ~ 15.9 h       |
|                  | 40 tok                          | ~ 21.3 h       |
|                  | 50 tok                          | ~ 26.6 h       |
|                  | 64 tok (your synthetic doc len) | ~ 34 h         |

---

### 🧩 Server configuration to record

```bash
docker run --name tei-gemma --gpus all -p 8080:80 \
  -e HF_TOKEN=... \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -e TOKENIZATION_WORKERS=$(nproc) \
  -v /srv/tei-cache:/data \
  ghcr.io/huggingface/text-embeddings-inference:89-1.8 \
    --model-id google/embeddinggemma-300m \
    --dtype float32 \
    --pooling mean \
    --max-batch-tokens 32768 \
    --max-client-batch-size 256 \
    --max-concurrent-requests 128 \
    --auto-truncate
```

---

### 🧾 Saved “best settings” block

```yaml
embeddinggemma_300m_fp32_4090_best:
  server:
    dtype: float32
    max_batch_tokens: 32768
    max_client_batch_size: 256
    max_concurrent_requests: 128
    pooling: mean
  client:
    batch: 32
    concurrency: 96
    avg_tok: 64
    std_tok: 48
    trial_docs: 20000
    trial_secs: 20
    warmup_secs: 3
  performance:
    docs_per_s: 512
    toks_per_s: 209175
    p50_s: 49.0
    p95_s: 55.2
    errors: 0
  estimated_runtime_400M:
    avg_tok_15: 8h
    avg_tok_30: 16h
    avg_tok_64: 34h
```

---

Next steps:

* Use this as your **1-GPU baseline**.
* For 2×/4× 4090 tests, double or quadruple the containers with the same per-GPU settings and run the same client grid with round-robin URLs to measure scaling efficiency.

---

### 🚀 Multi-GPU Scaling Results

The following results were achieved using the corrected `autotune_embed.py` script with client-side round-robin load balancing.

| GPUs | Throughput (tokens/sec) | Total Time | Scaling Factor |
| :--- | :--- | :--- | :--- |
| 1    | 24,375                  | 171.63s    | 1.00x          |
| 2    | 36,076                  | 121.21s    | 1.48x          |
| 4    | 69,912                  | 89.78s     | 2.87x          |

The optimal configuration for the four-GPU setup was found to be `BATCH=32` and `CONC=128`.
