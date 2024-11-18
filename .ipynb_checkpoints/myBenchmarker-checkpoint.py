import os
import sqlite3
import myWalksat as mw
import multiprocessing
from multiprocessing import Process, Queue
import logging

# Copyright Daniel Espinosa Gonzalez
# October 2024
# UCSB Strukov Lab

logging.basicConfig(
    filename='myBenchmarkerSubsetIterations.log',
    level=logging.INFO,                
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_algorithm(filename, filepath, mode, max_tries, max_flips, probability, result_queue):
    try:
        avg_flips, prob_s, std_flips, tts_99, avg_loops, std_loops, avg_tries, std_tries = mw.API(
            filepath, max_tries, max_flips, probability, mode)
        result = (filename, mode, avg_flips, prob_s, std_flips, tts_99, avg_loops, std_loops, avg_tries, std_tries)
        logging.info(f"Successfully processed {filename} with mode {mode}")
    except Exception as e:
        logging.error(f"Error processing {filename} with mode {mode}: {e}")
        result = (filename, mode, None, None, None, None, None, None, None, None)
    result_queue.put(result)

def batch(iterable, n=1):
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx:min(ndx + n, length)]

def main():
    directory = "/home/dae/SatExperiments/juniper/BenchmarkSubsetPaper/"
    if not os.path.isdir(directory):
        print(f"Directory {directory} does not exist.")
        return

    # Old: BenchmarkSubset // now iterations included (extra stats)
    db_file = "BenchmarkSubsetIterations.db"
    modes = ["walksat", "coloringA1_heuristic0", "coloringA1_heuristic1",
             "coloringA1_heuristic2", "coloringA1_heuristic3"]
    max_tries = 10
    max_flips = 100000
    probability = 0.5

    logging.info(f"Starting processing for directory: {directory}")

    # Prepare arguments for processing
    args_list = []
    files = [f for f in os.listdir(directory) if f.endswith('.cnf')]
    for filename in files:
        filepath = os.path.join(directory, filename)
        for mode in modes:
            args_list.append((filename, filepath, mode, max_tries, max_flips, probability))

    # Connect to the database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    try:
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS BenchmarkSubsetIterations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            algorithmname TEXT,
            avg_flips REAL,
            prob_s REAL,
            std_flips REAL,
            tts_99 REAL,
            avg_loops REAL,
            std_loops REAL,
            avg_tries REAL,
            std_tries REAL
        )
        ''')
        conn.commit()
        logging.info("Table BenchmarkSubsetIterations created or already exists.")
    
    except:
        logging.error(f"Error connecting to database")
        

        

    batch_size = 4  

    # Process tasks in batches
    for batch_args in batch(args_list, batch_size):
        logging.info(f"Starting batch with {len(batch_args)} files")
        result_queue = multiprocessing.Queue()
        processes = []

        # Start processes for the current batch
        for args in batch_args:
            filename, filepath, mode, max_tries, max_flips, probability = args
            logging.info(f"Starting process for file {filename} with mode {mode}")
            p = Process(target=process_algorithm, args=(filename, filepath, mode, max_tries, max_flips, probability, result_queue))
            processes.append(p)
            p.start()

        # Collect results
        results = []
        for _ in processes:
            result = result_queue.get()
            results.append(result)

        # Wait for processes to finish
        for p in processes:
            p.join()

        # Insert results into the database
        for result in results:
            cursor.execute('''
                INSERT INTO BenchmarkSubsetIterations (filename, algorithmname, avg_flips, prob_s, std_flips, tts_99, avg_loops, std_loops, avg_tries, std_tries)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', result)
            conn.commit()
            logging.info("Batch processed and results inserted into the database.")

    conn.close()
    logging.info("Processing complete. Database connection closed. Results in BenchmarkSubsetPaper.db")

if __name__ == "__main__":
    main()
