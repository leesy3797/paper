import multiprocessing
import asyncio
import argparse
import os
import time

from eval_pipeline import mainworkflow

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=str, required=True)
parser.add_argument('--baseline', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()

MAX_CONCURRENT_TASKS = 25

# Set log directory
LOG_DIR = f"{args.benchmark}/log/{args.baseline}/{args.model}"
COMPLETED_FILE = f"{LOG_DIR}/completed_ids.txt"
FAILED_FILE = f"{LOG_DIR}/failed_ids.txt"
TIME_LOGGING_FILE = f"{LOG_DIR}/execution_time.txt"

# Create log directory
os.makedirs(LOG_DIR, exist_ok=True)

def load_ids(filename):
    """Load ID list from file"""
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return set(map(int, f.read().splitlines()))
    return set()

def save_id(filename, test_sample_id):
    """Save ID to file"""
    with open(filename, "a") as f:
        f.write(f"{test_sample_id}\n")

def remove_id(filename, test_sample_id):
    """Remove specific ID from file"""
    if os.path.exists(filename):
        ids = load_ids(filename)
        if test_sample_id in ids:
            ids.remove(test_sample_id)
            with open(filename, "w") as f:
                for i in ids:
                    f.write(f"{i}\n")

def worker(test_sample_id):
    """Worker function to process each sample ID"""
    print(f"Processing: {test_sample_id}")
    try:
        asyncio.run(mainworkflow(test_sample_id=test_sample_id, benchmark=args.benchmark, baseline=args.baseline, model=args.model))
        save_id(COMPLETED_FILE, test_sample_id)  # Save to completed file on success
        remove_id(FAILED_FILE, test_sample_id)  # Remove from failed file on success
    except Exception as e:
        print(f"⚠️ Failed: {test_sample_id}, Error: {e}")
        save_id(FAILED_FILE, test_sample_id)  # Save to failed file on failure

def main():
    # Set sample ID list based on benchmark
    if args.benchmark == 'matplotbench':
        test_sample_ids = list(range(1, 101))  # 1~100
        # test_sample_ids = [5, 90] 
    elif args.benchmark == 'viseval':
        test_sample_ids = list(range(0, 1115))  # 0~1114
    else:
        raise ValueError("Invalid benchmark type")

    # Load previously completed IDs
    completed_ids = load_ids(COMPLETED_FILE)
    failed_ids = load_ids(FAILED_FILE)

    # Filter IDs to be processed (excluding already completed ones)
    remaining_ids = [i for i in test_sample_ids if i not in completed_ids]

    if not remaining_ids:
        print("✅ All test_sample_ids have already been processed.")
        return

    print(f"🔄 Running {len(remaining_ids)} test cases (including {len(failed_ids)} retries)...")
    start_time = time.time()

    # Create and execute multiprocessing pool
    with multiprocessing.Pool(processes=MAX_CONCURRENT_TASKS) as pool:
        pool.map(worker, remaining_ids)

    end_time = time.time()
    # 총 걸린 시간 계산
    elapsed_time = end_time - start_time

    save_id(TIME_LOGGING_FILE, elapsed_time)
    
    print(f"Total Processed Time : {elapsed_time:.6f}초")

if __name__ == "__main__":
    main()