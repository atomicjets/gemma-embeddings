import asyncio
import csv
import os
import random
import statistics
import sys
import time
from collections import defaultdict

import aiohttp
import numpy as np
from transformers import AutoTokenizer

# ----------------------- Config via env -----------------------
ACCOUNT_ID = os.getenv("ACCOUNT_ID")
API_TOKEN = os.getenv("API_TOKEN")
URL = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/@cf/google/embeddinggemma-300m"

MODEL_ID = os.getenv("MODEL_ID", "google/embeddinggemma-300m")

# Text length profile (approx tokens per doc)
AVG_TOK = int(os.getenv("AVG_TOK", "64"))
STD_TOK = int(os.getenv("STD_TOK", "48"))

# Search space
BATCH_GRID = [int(x) for x in os.getenv("BATCH_GRID", "8,16,32").split(",")]
CONC_GRID = [int(x) for x in os.getenv("CONC_GRID", "16,32,64").split(",")]

# Trial control
TRIAL_DOCS = int(os.getenv("TRIAL_DOCS", "20000"))  # per trial
TRIAL_SECS = float(os.getenv("TRIAL_SECS", "20"))  # max seconds per trial
WARMUP_SECS = float(os.getenv("WARMUP_SECS", "3"))
P95_TARGET = float(os.getenv("P95_TARGET", "10.0"))  # seconds
REPORT_EVERY = float(os.getenv("REPORT_EVERY", "2"))

# Output
CSV_PATH = os.getenv("CSV_PATH", "autotune_cloudflare_results.csv")
# --------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
random.seed(42)

def synth_doc(avg: int = AVG_TOK, std: int = STD_TOK) -> str:
    """Synthetically generate a doc with ~avg tokens using a crude 0.75 tok/word heuristic."""
    words = max(4, int((avg + random.randint(-std, std)) / 0.75))
    return " ".join([f"tok{random.randint(1,9999)}" for _ in range(words)])

def gen_batches(total_docs: int, batch_size: int) -> list[list[str]]:
    """Generate batches of documents."""
    docs = [synth_doc() for _ in range(total_docs)]
    return [docs[i : i + batch_size] for i in range(0, len(docs), batch_size)]


async def run_trial(batch_size: int, conc: int) -> dict:
    """Run one (BATCH, CONC) trial and return metrics dict."""
    q: asyncio.Queue[list[str] | None] = asyncio.Queue(maxsize=conc * 2)
    lat_ms: list[float] = []
    tok_ctr: list[int] = []
    doc_ctr: list[int] = []
    err_ctr: list[int] = []

    stop_time = time.perf_counter() + TRIAL_SECS
    connector = aiohttp.TCPConnector(limit=conc * 4, enable_cleanup_closed=True)
    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:

        async def worker() -> None:
            while True:
                batch = await q.get()
                if batch is None:
                    q.task_done()
                    return

                # token estimate
                toks = tokenizer(
                    batch, add_special_tokens=False, truncation=True, max_length=2048
                )["input_ids"]
                tok_count = sum(len(t) for t in toks)
                payload = {"text": batch}

                t0 = time.perf_counter()
                try:
                    async with session.post(URL, json=payload, timeout=60) as r:
                        if r.status != 200:
                            err_ctr.append(1)
                            error_text = await r.text()
                            print(f"Error: status={r.status}, body={error_text[:200]}")
                        else:
                            await r.read()  # drain to keep pool healthy
                except Exception as e:
                    err_ctr.append(1)
                    print(f"Request failed: {e}")
                t1 = time.perf_counter()

                lat_ms.append((t1 - t0) * 1000.0)
                tok_ctr.append(tok_count)
                doc_ctr.append(len(batch))
                q.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(conc)]

        # Producer: keep feeding until stop_time or we exhaust TRIAL_DOCS
        produced = 0
        trial_start_time = time.perf_counter()
        for batch in gen_batches(TRIAL_DOCS, batch_size):
            if time.perf_counter() >= stop_time:
                break
            await q.put(batch)
            produced += len(batch)
        
        # Wait for all items to be processed
        await q.join()

        # poison pills
        for _ in workers:
            await q.put(None)

        # Reporter (optional): lightweight progress
        async def reporter() -> None:
            last_docs = 0
            last_tokens = 0
            last_t = time.perf_counter()

            while any(not w.done() for w in workers):
                await asyncio.sleep(REPORT_EVERY)
                now = time.perf_counter()
                docs = sum(doc_ctr)
                toks = sum(tok_ctr)
                interval = max(1e-6, now - last_t)
                docs_s = (docs - last_docs) / interval
                toks_s = (toks - last_tokens) / interval

                # Minimal progress line
                print(
                    f"[B={batch_size:>3} C={conc:>3}] "
                    f"docs/s={docs_s:>8.0f} tok/s={toks_s:>9.0f} "
                    f"lat_p50={(statistics.median(lat_ms) if lat_ms else 0)/1000:>4.2f}s "
                    f"errs={sum(err_ctr)}"
                )

                last_docs, last_tokens, last_t = docs, toks, now

        reporter_task = asyncio.create_task(reporter())
        await asyncio.gather(*workers)
        trial_end_time = time.perf_counter()
        reporter_task.cancel()

    # compute metrics (discard initial warmup window if long test)
    if len(lat_ms) > 5 and WARMUP_SECS > 0:
        # approximate warmup trim: drop earliest samples within WARMUP_SECS window
        # Coarse trim: remove first N items
        trim = min(int(len(lat_ms) * (WARMUP_SECS / max(TRIAL_SECS, 1))), len(lat_ms) // 4)
        lat_use = lat_ms[trim:]
    else:
        lat_use = lat_ms

    docs = sum(doc_ctr)
    toks = sum(tok_ctr)
    dur = trial_end_time - trial_start_time
    docs_s = docs / max(dur, 1e-6)
    toks_s = toks / max(dur, 1e-6)
    p50 = statistics.median(lat_use) / 1000 if lat_use else 0.0
    p95 = float(np.percentile(lat_use, 95)) / 1000 if lat_use else 0.0

    return {
        "batch": batch_size,
        "conc": conc,
        "docs": docs,
        "tokens": toks,
        "docs_s": docs_s,
        "toks_s": toks_s,
        "p50_s": p50,
        "p95_s": p95,
        "errors": sum(err_ctr),
        "dur": dur,
    }


async def main() -> None:
    if not ACCOUNT_ID or not API_TOKEN:
        print("Error: ACCOUNT_ID and API_TOKEN environment variables must be set.")
        sys.exit(1)

    results: list[dict] = []
    best: dict | None = None

    for b in BATCH_GRID:
        for c in CONC_GRID:
            print(f"\n=== Trial BATCH={b}, CONC={c} ===")
            metrics = await run_trial(b, c)
            results.append(metrics)

            ok = (metrics["p95_s"] <= P95_TARGET) and (metrics["errors"] == 0)
            if ok and (best is None or metrics["toks_s"] > best["toks_s"]):
                best = metrics

            print(
                "Result: "
                f"docs/s={metrics['docs_s']:,.0f} tok/s={metrics['toks_s']:,.0f} "
                f"p50={metrics['p50_s']:.2f}s p95={metrics['p95_s']:.2f}s "
                f"errors={metrics['errors']} "
                f"total_time={metrics['dur']:.2f}s"
            )

    # Write CSV
    if results:
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)

        print(f"\n=== All trials written to {CSV_PATH} ===")

    if best:
        print(
            f"🏆 Best under p95≤{P95_TARGET:.1f}s: "
            f"BATCH={best['batch']} CONC={best['conc']} "
            f"tok/s={best['toks_s']:,.0f} p95={best['p95_s']:.2f}s p50={best['p50_s']:.2f}s"
        )
    else:
        print(
            f"❗ No combo met p95≤{P95_TARGET:.1f}s without errors. "
            f"Consider lowering BATCH or server --max-batch-tokens."
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
