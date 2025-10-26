#!/usr/bin/env bash
set -euo pipefail
env >> /etc/environment || true

# --- Defaults (overridable via env) ---
MODEL_ID="${MODEL_ID:-google/embeddinggemma-300m}"
DTYPE="${DTYPE:-float32}"
POOLING="${POOLING:-mean}"
MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-32768}"
MAX_CLIENT_BATCH_SIZE="${MAX_CLIENT_BATCH_SIZE:-256}"
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-128}"
AUTO_TRUNCATE="${AUTO_TRUNCATE:-1}"
TOKENIZATION_WORKERS="${TOKENIZATION_WORKERS:-$(nproc)}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# --- Locate TEI binary ---
BIN="$(command -v text-embeddings-router || command -v text-embeddings-inference || true)"
[ -z "$BIN" ] && { echo "TEI binary not found"; sleep infinity; }

# --- Detect GPUs (cap at 4) ---
NGPU="$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)"; [ "$NGPU" -gt 4 ] && NGPU=4
echo "GPUs: $NGPU | Model: $MODEL_ID | TW: $TOKENIZATION_WORKERS"

# --- Helper: start one instance safely (no unbound vars) ---
start_one() {
  local gpu port log
  gpu="$1"
  port="$2"
  log="/var/log/tei-gpu${gpu}.log"
  echo "Starting TEI on GPU ${gpu} -> 127.0.0.1:${port}"
  CUDA_VISIBLE_DEVICES="$gpu" "$BIN" \
    --model-id "$MODEL_ID" \
    --dtype "$DTYPE" \
    --pooling "$POOLING" \
    --max-batch-tokens "$MAX_BATCH_TOKENS" \
    --max-client-batch-size "$MAX_CLIENT_BATCH_SIZE" \
    --max-concurrent-requests "$MAX_CONCURRENT_REQUESTS" \
    --tokenization-workers "$TOKENIZATION_WORKERS" \
    ${AUTO_TRUNCATE:+--auto-truncate} \
    --hostname 127.0.0.1 \
    --port "$port" >"$log" 2>&1 &
}

# --- Staggered starts to avoid HF cache lock collisions ---
BASE=18080
for i in $(seq 0 $((NGPU-1))); do
  start_one "$i" $((BASE+i))
  sleep 20
done

# --- Stream logs so Vast shows progress ---
tail -F /var/log/tei-gpu*.log
