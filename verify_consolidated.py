import argparse
import logging
import pyarrow.parquet as pq

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def verify_file(filepath):
    """
    Reads a Parquet file and prints its metadata.
    """
    try:
        table = pq.read_table(filepath)
        logging.info(f"Successfully read file: {filepath}")
        logging.info(f"Number of rows: {table.num_rows}")
        logging.info(f"Schema:\n{table.schema}")
        
        # You can also inspect a small sample of the data
        # logging.info(f"Sample data:\n{table.slice(0, 5).to_pandas()}")

    except Exception as e:
        logging.error(f"Failed to read or verify Parquet file {filepath}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify a consolidated Parquet file.")
    parser.add_argument(
        "filepath",
        type=str,
        help="Path to the Parquet file to verify."
    )
    args = parser.parse_args()
    verify_file(args.filepath)
