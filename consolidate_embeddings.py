import os
import argparse
import logging
import pyarrow.parquet as pq
import pyarrow as pa
from glob import glob
from multiprocessing import Pool, cpu_count

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_chunk(args):
    """Processes a single chunk of files."""
    chunk_num, file_chunk, output_dir = args
    if not file_chunk:
        return None

    logging.info(f"Processing chunk {chunk_num + 1}: {len(file_chunk)} files...")
    try:
        tables = [pq.read_table(f) for f in file_chunk]
        if not tables:
            return None

        consolidated_table = pa.concat_tables(tables)
        output_filename = f"consolidated-part-{chunk_num:05d}.parquet"
        output_path = os.path.join(output_dir, output_filename)

        pq.write_table(
            consolidated_table,
            output_path,
            row_group_size=65536,
            compression='zstd'
        )
        logging.info(f"Successfully wrote consolidated file: {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Failed to process chunk {chunk_num + 1}: {e}")
        return None

def consolidate_files(
    source_dirs,
    output_dir,
    files_per_chunk=10000,
    num_workers=None
):
    """
    Consolidates small Parquet files from multiple source directories into larger ones using multiprocessing.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    all_files = []
    for source_dir in source_dirs:
        files = glob(os.path.join(source_dir, '*.parquet'))
        logging.info(f"Found {len(files)} files in {source_dir}.")
        all_files.extend(files)
    
    if not all_files:
        logging.warning(f"No Parquet files found in any source directories. Exiting.")
        return
    
    all_files.sort()

    logging.info(f"Found {len(all_files)} files to consolidate.")

    chunks = [
        (i, all_files[i*files_per_chunk:(i+1)*files_per_chunk], output_dir)
        for i in range((len(all_files) + files_per_chunk - 1) // files_per_chunk)
    ]

    if not num_workers:
        num_workers = cpu_count()
    
    logging.info(f"Starting consolidation with {num_workers} worker processes.")

    with Pool(processes=num_workers) as pool:
        results = pool.map(process_chunk, chunks)

    successful_files = [r for r in results if r]
    logging.info(f"Consolidation process complete. {len(successful_files)} files created.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate small Parquet files from one or more source directories.")
    parser.add_argument(
        "--source-dirs",
        type=str,
        nargs='+',
        required=True,
        help="One or more source directories containing the small Parquet files."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/sdb_mount_point/embeddings/consolidated",
        help="Directory to save the consolidated Parquet files."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Number of small files to process in each consolidation chunk."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes to use. Defaults to the number of CPU cores."
    )

    args = parser.parse_args()

    consolidate_files(
        source_dirs=args.source_dirs,
        output_dir=args.output_dir,
        files_per_chunk=args.chunk_size,
        num_workers=args.num_workers
    )
