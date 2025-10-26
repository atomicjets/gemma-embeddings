import asyncio, aiohttp, time, statistics, os, random, numpy as np
from transformers import AutoTokenizer

TEI_URL = os.getenv("TEI_URL", "http://localhost:8080/embed")
MODEL_ID = os.getenv("MODEL_ID", "google/embeddinggemma-300m")

NUM_DOCS = int(os.getenv("NUM_DOCS", "200000"))
AVG_TOK  = int(os.getenv("AVG_TOK",  "64"))
STD_TOK  = int(os.getenv("STD_TOK",  "16"))
BATCH    = int(os.getenv("BATCH",    "256"))
CONC     = int(os.getenv("CONC",     "8"))
TIMEOUT  = int(os.getenv("TIMEOUT",  "60"))
REPORT_EVERY = float(os.getenv("REPORT_EVERY", "2"))

NORMALISE_LOCAL = bool(int(os.getenv("NORMALISE_LOCAL", "0")))  # set 1 if your model’s pooling config doesn’t normalise

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def synth_doc(avg=AVG_TOK, std=STD_TOK):
    # crude synthetic text roughly avg tokens long (~0.75 tok/word)
    words = max(4, int((avg + random.randint(-std, std)) / 0.75))
    return " ".join([f"lorem_{random.randint(1,9999)}" for _ in range(words)])

def make_batch(n=BATCH):
    return [synth_doc() for _ in range(n)]

async def worker(name, session, q, lat_ms, tok_ctr, doc_ctr, err_ctr):
    while True:
        batch = await q.get()
        if batch is None:
            q.task_done(); return

        # token estimate for throughput calc
        toks = tokenizer(batch, add_special_tokens=False, truncation=True, max_length=2048)["input_ids"]
        token_count = sum(len(t) for t in toks)

        payload = {"inputs": batch}
        t0 = time.perf_counter()
        try:
            async with session.post(TEI_URL, json=payload, timeout=TIMEOUT) as r:
                if r.status != 200:
                    err_ctr.append(1)
                    await r.read()
                else:
                    data = await r.json()
                    # optionally normalise client-side (shouldn’t be needed for embeddinggemma)
                    if NORMALISE_LOCAL and isinstance(data, dict) and "embeddings" in data:
                        arr = np.array(data["embeddings"], dtype=np.float32)
                        nrm = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
                        arr = arr / nrm
        except Exception:
            err_ctr.append(1)
        t1 = time.perf_counter()

        lat_ms.append((t1 - t0) * 1000.0)
        tok_ctr.append(token_count)
        doc_ctr.append(len(batch))
        q.task_done()

async def main():
    total_batches = NUM_DOCS // BATCH + (1 if NUM_DOCS % BATCH else 0)
    q = asyncio.Queue(maxsize=CONC * 2)

    lat_ms, tok_ctr, doc_ctr, err_ctr = [], [], [], []

    async with aiohttp.ClientSession() as session:
        workers = [asyncio.create_task(worker(f"w{i}", session, q, lat_ms, tok_ctr, doc_ctr, err_ctr))
                   for i in range(CONC)]

        async def producer():
            for _ in range(total_batches):
                await q.put(make_batch())
            for _ in workers:
                await q.put(None)

        async def reporter():
            last_docs = 0
            last_tokens = 0
            last_time = time.perf_counter()
            start = last_time
            while any(not w.done() for w in workers):
                await asyncio.sleep(REPORT_EVERY)
                now = time.perf_counter()
                docs = sum(doc_ctr)
                toks = sum(tok_ctr)
                interval = max(1e-6, now - last_time)
                docs_s = (docs - last_docs) / interval
                toks_s = (toks - last_tokens) / interval
                if lat_ms:
                    window = lat_ms[-min(2000, len(lat_ms)):]
                    p50 = statistics.median(window)
                    p95 = float(np.percentile(window, 95))
                else:
                    p50 = p95 = 0.0
                print(f"[{now-start:6.1f}s] docs/s={docs_s:,.0f}  tok/s={toks_s:,.0f}  "
                      f"p50={p50:.1f}ms  p95={p95:.1f}ms  errors={sum(err_ctr)}")
                last_docs, last_tokens, last_time = docs, toks, now

        await asyncio.gather(producer(), reporter(), *workers)

        dur = time.perf_counter() - (time.perf_counter() - 0)  # just for formatting parity
        docs = sum(doc_ctr); toks = sum(tok_ctr)
        all_p50 = statistics.median(lat_ms) if lat_ms else 0.0
        all_p95 = float(np.percentile(lat_ms, 95)) if lat_ms else 0.0
        all_p99 = float(np.percentile(lat_ms, 99)) if lat_ms else 0.0
        print("\n=== Summary ===")
        print(f"Total docs: {docs:,}")
        print(f"Total toks: {toks:,}")
        # Compute true duration based on first & last timestamps of latencies
        # (simpler: wall clock)
        # If you prefer, wrap start/stop time externally.
        print(f"Latency p50={all_p50:.1f}ms  p95={all_p95:.1f}ms  p99={all_p99:.1f}ms")
        print(f"Errors: {sum(err_ctr)}")

if __name__ == "__main__":
    asyncio.run(main())
