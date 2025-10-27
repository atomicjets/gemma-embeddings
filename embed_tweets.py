import os
import re
import time
import logging
import argparse
import shutil
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
import pyarrow as pa
import pyarrow.parquet as pq
import asyncio
import aiohttp
from itertools import cycle

# --- Configuration ---
# Basic Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Text Cleaning ---
def clean_tweet_text(text):
    """
    Cleans tweet text by removing URLs, @ symbols, and # symbols,
    and normalizing whitespace.
    """
    if not text:
        return ""
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove @ and # symbols but keep the text
    text = text.replace('@', '').replace('#', '')
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Main Embedding Logic ---
def process_embeddings(
    mongo_uri,
    db_name,
    collection_name,
    tei_urls,
    tmp_dir,
    final_dir,
    batch_size=32,
    concurrency=192,
    max_retries=3
):
    """
    Main function to fetch tweets, generate embeddings, and save them concurrently.
    """
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    async def main():
        queue = asyncio.Queue(maxsize=concurrency * 2)
        url_cycler = cycle(tei_urls)
        
        # Performance counters
        total_docs_processed = 0
        start_time = time.time()

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

        async def worker(session):
            nonlocal total_docs_processed
            while True:
                batch_docs = await queue.get()
                if batch_docs is None:
                    queue.task_done()
                    return

                ids = [doc['_id'] for doc in batch_docs]
                texts_to_embed = [clean_tweet_text(doc.get('text', '')) for doc in batch_docs]

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
                except Exception as e:
                    logging.error(f"Failed to write/move Parquet for batch starting with {ids[0]}: {e}")
                    queue.task_done()
                    continue

                operations = [UpdateOne({"_id": doc_id}, {"$set": {"gemma_embedded_768": True}}) for doc_id in ids]
                try:
                    collection.bulk_write(operations, ordered=False)
                    total_docs_processed += len(ids)
                except Exception as e:
                    logging.error(f"MongoDB update failed: {e}")

                queue.task_done()

        # --- Main async execution ---
        reporter_task = asyncio.create_task(reporter())
        
        async with aiohttp.ClientSession() as session:
            workers = [asyncio.create_task(worker(session)) for _ in range(concurrency)]

            # Producer loop
            while True:
                query = {"lang": "en", "classified": True, "retweeted": None, "gemma_embedded_768": None}
                # Fetch enough for the whole worker pool to get busy
                docs_to_fetch = batch_size * concurrency 
                docs = list(collection.find(query, {"_id": 1, "text": 1}).limit(docs_to_fetch))

                if not docs:
                    logging.info("Producer: No more documents to process.")
                    break
                
                for i in range(0, len(docs), batch_size):
                    await queue.put(docs[i:i+batch_size])
                
                # If we fetched fewer docs than requested, we are at the end
                if len(docs) < docs_to_fetch:
                    break

            await queue.join()
            for _ in workers: await queue.put(None)
            await asyncio.gather(*workers)

        reporter_task.cancel()

    asyncio.run(main())
    client.close()
    logging.info("Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and store tweet embeddings.")
    parser.add_argument("--mongo-uri", type=str, default="mongodb://localhost:27017/", help="MongoDB connection URI.")
    parser.add_argument("--db-name", type=str, default="twitter_data", help="MongoDB database name.")
    parser.add_argument("--collection", type=str, required=True, help="MongoDB collection to process (e.g., 'tweets' or 'tweets_usa').")
    parser.add_argument("--tei-urls", type=str, required=True, help="Comma-separated TEI endpoint URLs.")
    parser.add_argument("--tmp-dir", type=str, default="/mnt/raid1/embeddings/tmp", help="Temporary directory for Parquet files.")
    parser.add_argument("--final-dir", type=str, default="/mnt/sdb_mount_point/embeddings/shards", help="Final directory for Parquet files.")
    parser.add_argument("--batch-size", type=int, default=32, help="Number of documents to process in a batch.")
    parser.add_argument("--concurrency", type=int, default=192, help="Number of concurrent requests.")

    args = parser.parse_args()
    
    tei_endpoints = [url.strip() for url in args.tei_urls.split(',')]

    process_embeddings(
        mongo_uri=args.mongo_uri,
        db_name=args.db_name,
        collection_name=args.collection,
        tei_urls=tei_endpoints,
        tmp_dir=args.tmp_dir,
        final_dir=args.final_dir,
        batch_size=args.batch_size,
        concurrency=args.concurrency
    )
