import os
import json
import time
import logging
import argparse
import shutil
import re
import pyarrow as pa
import pyarrow.parquet as pq
import asyncio
import aiohttp
from itertools import cycle, islice

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Text Cleaning ---
def clean_tweet_text(text):
    if not text:
        return ""
    text = re.sub(r'https?://\S+', '', text)
    text = text.replace('@', '').replace('#', '')
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:1000]

# --- Main Embedding Logic ---
async def process_embeddings_from_file(
    input_file,
    tei_urls,
    tmp_dir,
    final_dir,
    progress_dir,
    batch_size=32,
    concurrency=128,
    max_retries=3
):
    url_cycler = cycle(tei_urls)
    queue = asyncio.Queue(maxsize=concurrency * 2)
    total_docs_processed = 0
    start_time = time.time()

    # --- Reporter Coroutine ---
    async def reporter():
        nonlocal total_docs_processed
        last_log_time = start_time
        docs_in_last_interval = 0
        while True:
            await asyncio.sleep(5)
            current_docs = total_docs_processed
            current_time = time.time()
            elapsed_interval = current_time - last_log_time
            docs_this_interval = current_docs - docs_in_last_interval
            docs_per_sec_interval = docs_this_interval / elapsed_interval
            elapsed_total = current_time - start_time
            docs_per_sec_total = current_docs / elapsed_total if elapsed_total > 0 else 0
            logging.info(
                f"Inst. speed: {docs_per_sec_interval:.2f} docs/s | "
                f"Avg speed: {docs_per_sec_total:.2f} docs/s | "
                f"Total processed: {current_docs}"
            )
            last_log_time = current_time
            docs_in_last_interval = current_docs

    # --- Worker Coroutine ---
    async def worker(session):
        nonlocal total_docs_processed
        while True:
            batch_docs = await queue.get()
            if batch_docs is None:
                queue.task_done()
                return

            valid_docs = []
            for doc in batch_docs:
                text = doc.get('text')
                if text:
                    cleaned_text = clean_tweet_text(text)
                    if cleaned_text:
                        valid_docs.append({'_id': doc['_id'], 'text': cleaned_text})
            
            if not valid_docs:
                queue.task_done()
                continue

            # Handle both string and nested $oid formats for _id
            ids = []
            for doc in valid_docs:
                if isinstance(doc['_id'], dict) and '$oid' in doc['_id']:
                    ids.append(doc['_id']['$oid'])
                else:
                    ids.append(str(doc['_id']))

            texts_to_embed = [doc['text'] for doc in valid_docs]
            embeddings = []
            for i in range(max_retries):
                try:
                    tei_url = next(url_cycler)
                    async with session.post(
                        tei_url, json={"inputs": texts_to_embed, "truncate": True}, timeout=120
                    ) as response:
                        response.raise_for_status()
                        embeddings = await response.json()
                        break
                except Exception as e:
                    logging.warning(f"TEI request failed (attempt {i+1}/{max_retries}): {e}")
                    if i == max_retries - 1: logging.error("Max retries reached, skipping batch.")
                    await asyncio.sleep(5)
            
            if not embeddings:
                queue.task_done()
                continue

            try:
                processed_embeddings = [emb[0] if isinstance(emb, list) and len(emb) == 1 and isinstance(emb[0], list) else emb for emb in embeddings]
                arrow_table = pa.Table.from_pydict({
                    'id': pa.array(ids, type=pa.string()),
                    'embedding': pa.array(processed_embeddings, type=pa.list_(pa.float32()))
                })
                part_filename = f"part-{ids[0]}-{ids[-1]}.parquet"
                tmp_path = os.path.join(tmp_dir, part_filename)
                final_path = os.path.join(final_dir, part_filename)
                pq.write_table(arrow_table, tmp_path, compression='zstd')
                shutil.move(tmp_path, final_path)
                total_docs_processed += len(ids)
            except Exception as e:
                logging.error(f"Failed to write/move Parquet for batch starting with {ids[0]}: {e}")
            
            queue.task_done()

    # --- Main async execution ---
    reporter_task = asyncio.create_task(reporter())
    
    async with aiohttp.ClientSession() as session:
        workers = [asyncio.create_task(worker(session)) for _ in range(concurrency)]

        # --- Producer Logic (from file) ---
        chunk_size = 1000000 # Read 1M lines at a time for resumability
        with open(input_file, 'r') as f:
            chunk_num = 0
            while True:
                progress_file = os.path.join(progress_dir, f"chunk-{chunk_num}.done")
                if os.path.exists(progress_file):
                    logging.info(f"Skipping already processed chunk {chunk_num}")
                    # Fast-forward the file handle
                    for _ in range(chunk_size):
                        next(f, None)
                    chunk_num += 1
                    continue

                lines = list(islice(f, chunk_size))
                if not lines:
                    break

                batch = []
                for line in lines:
                    try:
                        record = json.loads(line)
                        if '_id' in record and 'text' in record:
                            batch.append(record)
                        if len(batch) >= batch_size:
                            await queue.put(batch)
                            batch = []
                    except json.JSONDecodeError:
                        logging.warning(f"Skipping malformed line: {line.strip()}")
                
                if batch:
                    await queue.put(batch)
                
                # Mark chunk as done
                with open(progress_file, 'w') as pf:
                    pf.write('done')
                chunk_num += 1

        await queue.join()
        for _ in workers: await queue.put(None)
        await asyncio.gather(*workers)

    reporter_task.cancel()
    logging.info("Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tweet embeddings from a file.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the line-delimited JSON input file.")
    parser.add_argument("--tei-urls", type=str, required=True, help="Comma-separated TEI endpoint URLs.")
    parser.add_argument("--tmp-dir", type=str, default="/mnt/raid1/embeddings/tmp", help="Temporary directory for Parquet files.")
    parser.add_argument("--final-dir", type=str, default="/mnt/sdb_mount_point/embeddings/shards", help="Final directory for Parquet files.")
    parser.add_argument("--progress-dir", type=str, default="/mnt/raid1/embeddings/progress", help="Directory to store progress markers.")
    parser.add_argument("--batch-size", type=int, default=32, help="Number of documents to process in a batch.")
    parser.add_argument("--concurrency", type=int, default=128, help="Number of concurrent requests.")

    args = parser.parse_args()
    
    if not os.path.exists(args.progress_dir):
        os.makedirs(args.progress_dir)

    tei_endpoints = [url.strip() for url in args.tei_urls.split(',')]

    asyncio.run(process_embeddings_from_file(
        input_file=args.input_file,
        tei_urls=tei_endpoints,
        tmp_dir=args.tmp_dir,
        final_dir=args.final_dir,
        progress_dir=args.progress_dir,
        batch_size=args.batch_size,
        concurrency=args.concurrency
    ))
