import os
import argparse
import logging
import pyarrow.parquet as pq
import pyarrow as pa
from glob import glob

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def consolidate_files(
    source_dir,
    output_dir,
    files_per_chunk=10000,
    target_file_size_mb=1024
):
    """
    Consolidates small Parquet files into larger ones.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    all_files = sorted(glob(os.path.join(source_dir, '*.parquet')))
    if not all_files:
        logging.warning(f"No Parquet files found in {source_dir}. Exiting.")
        return

    logging.info(f"Found {len(all_files)} files to consolidate.")

    output_chunk_num = 0
    for i in range(0, len(all_files), files_per_chunk):
        file_chunk = all_files[i:i + files_per_chunk]
        
        if not file_chunk:
            continue

        logging.info(f"Processing chunk {output_chunk_num + 1}: {len(file_chunk)} files...")

        try:
            tables = [pq.read_table(f) for f in file_chunk]
            if not tables:
                continue

            consolidated_table = pa.concat_tables(tables)

            output_filename = f"consolidated-part-{output_chunk_num:05d}.parquet"
            output_path = os.path.join(output_dir, output_filename)

            pq.write_table(
                consolidated_table,
                output_path,
                row_group_size=65536, # Good default for large files
                compression='zstd'
            )
            logging.info(f"Successfully wrote consolidated file: {output_path}")
            output_chunk_num += 1

        except Exception as e:
            logging.error(f"Failed to process a chunk of files: {e}")
            # Continue to the next chunk
            continue

    logging.info("Consolidation process complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate small Parquet files.")
    parser.add_argument(
        "--source-dir",
        type=str,
        default="/mnt/sdb_mount_point/embeddings/shards",
        help="Directory containing the small Parquet files."
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

    args = parser.parse_args()

    consolidate_files(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        files_per_chunk=args.chunk_size
    )
