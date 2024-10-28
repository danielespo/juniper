import os
import sqlite3
import myWalksat as mw
import multiprocessing
from multiprocessing import Process, Queue
import logging

# Copyright Daniel Espinosa Gonzalez
# October 2024
# UCSB Strukov Lab
# Modified to generate cumulative density function plots

logging.basicConfig(
    filename='myBenchmarkerSubsetIterationsCDF.log',
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
    db_file = "BenchmarkSubsetIterationsCDF.db"
    modes = ["walksat", "coloringA1_heuristic0", "coloringA1_heuristic1",
             "coloringA1_heuristic2", "coloringA1_heuristic3"]
    max_tries = 10
    max_flips = 10
    probability = 0.5

    logging.info(f"Starting processing for directory: {directory}")

    # Prepare arguments for processing
    args_list = []
    files = [f for f in os.listdir(directory) if f.endswith('.cnf')]
    for filename in files:
        filepath = os.path.join(directory, filename)
        for mode in modes:
            args_list.append((filename, filepath, mode, max_tries, max_flips, probability))

    max_flips = 20
    logging.info(f"Starting processing for directory: {directory}")

    # Prepare arguments for processing
    args_list = []
    files = [f for f in os.listdir(directory) if f.endswith('.cnf')]
    for filename in files:
        filepath = os.path.join(directory, filename)
        for mode in modes:
            args_list.append((filename, filepath, mode, max_tries, max_flips, probability))

    max_flips = 30
    logging.info(f"Starting processing for directory: {directory}")

    # Prepare arguments for processing
    args_list = []
    files = [f for f in os.listdir(directory) if f.endswith('.cnf')]
    for filename in files:
        filepath = os.path.join(directory, filename)
        for mode in modes:
            args_list.append((filename, filepath, mode, max_tries, max_flips, probability))

    max_flips = 50
    logging.info(f"Starting processing for directory: {directory}")

    # Prepare arguments for processing
    args_list = []
    files = [f for f in os.listdir(directory) if f.endswith('.cnf')]
    for filename in files:
        filepath = os.path.join(directory, filename)
        for mode in modes:
            args_list.append((filename, filepath, mode, max_tries, max_flips, probability))

    max_flips = 60
    logging.info(f"Starting processing for directory: {directory}")

    # Prepare arguments for processing
    args_list = []
    files = [f for f in os.listdir(directory) if f.endswith('.cnf')]
    for filename in files:
        filepath = os.path.join(directory, filename)
        for mode in modes:
            args_list.append((filename, filepath, mode, max_tries, max_flips, probability))


    max_flips = 70
    logging.info(f"Starting processing for directory: {directory}")

    # Prepare arguments for processing
    args_list = []
    files = [f for f in os.listdir(directory) if f.endswith('.cnf')]
    for filename in files:
        filepath = os.path.join(directory, filename)
        for mode in modes:
            args_list.append((filename, filepath, mode, max_tries, max_flips, probability))

    max_flips = 80
    logging.info(f"Starting processing for directory: {directory}")

    # Prepare arguments for processing
    args_list = []
    files = [f for f in os.listdir(directory) if f.endswith('.cnf')]
    for filename in files:
        filepath = os.path.join(directory, filename)
        for mode in modes:
            args_list.append((filename, filepath, mode, max_tries, max_flips, probability))

    max_flips = 90
    logging.info(f"Starting processing for directory: {directory}")

    # Prepare arguments for processing
    args_list = []
    files = [f for f in os.listdir(directory) if f.endswith('.cnf')]
    for filename in files:
        filepath = os.path.join(directory, filename)
        for mode in modes:
            args_list.append((filename, filepath, mode, max_tries, max_flips, probability))

    max_flips = 100
    logging.info(f"Starting processing for directory: {directory}")

    # Prepare arguments for processing
    args_list = []
    files = [f for f in os.listdir(directory) if f.endswith('.cnf')]
    for filename in files:
        filepath = os.path.join(directory, filename)
        for mode in modes:
            args_list.append((filename, filepath, mode, max_tries, max_flips, probability))

    max_flips = 500
    logging.info(f"Starting processing for directory: {directory}")

    # Prepare arguments for processing
    args_list = []
    files = [f for f in os.listdir(directory) if f.endswith('.cnf')]
    for filename in files:
        filepath = os.path.join(directory, filename)
        for mode in modes:
            args_list.append((filename, filepath, mode, max_tries, max_flips, probability))


    max_flips = 1000
    logging.info(f"Starting processing for directory: {directory}")

    # Prepare arguments for processing
    args_list = []
    files = [f for f in os.listdir(directory) if f.endswith('.cnf')]
    for filename in files:
        filepath = os.path.join(directory, filename)
        for mode in modes:
            args_list.append((filename, filepath, mode, max_tries, max_flips, probability))

    max_flips = 3000
    logging.info(f"Starting processing for directory: {directory}")

    # Prepare arguments for processing
    args_list = []
    files = [f for f in os.listdir(directory) if f.endswith('.cnf')]
    for filename in files:
        filepath = os.path.join(directory, filename)
        for mode in modes:
            args_list.append((filename, filepath, mode, max_tries, max_flips, probability))

    max_flips = 4000
    logging.info(f"Starting processing for directory: {directory}")

    # Prepare arguments for processing
    args_list = []
    files = [f for f in os.listdir(directory) if f.endswith('.cnf')]
    for filename in files:
        filepath = os.path.join(directory, filename)
        for mode in modes:
            args_list.append((filename, filepath, mode, max_tries, max_flips, probability))

    max_flips = 5000
    logging.info(f"Starting processing for directory: {directory}")

    # Prepare arguments for processing
    args_list = []
    files = [f for f in os.listdir(directory) if f.endswith('.cnf')]
    for filename in files:
        filepath = os.path.join(directory, filename)
        for mode in modes:
            args_list.append((filename, filepath, mode, max_tries, max_flips, probability))

    max_flips = 6000
    logging.info(f"Starting processing for directory: {directory}")

    # Prepare arguments for processing
    args_list = []
    files = [f for f in os.listdir(directory) if f.endswith('.cnf')]
    for filename in files:
        filepath = os.path.join(directory, filename)
        for mode in modes:
            args_list.append((filename, filepath, mode, max_tries, max_flips, probability))

    max_flips = 7000
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

    max_flips = 8000
    logging.info(f"Starting processing for directory: {directory}")

    # Prepare arguments for processing
    args_list = []
    files = [f for f in os.listdir(directory) if f.endswith('.cnf')]
    for filename in files:
        filepath = os.path.join(directory, filename)
        for mode in modes:
            args_list.append((filename, filepath, mode, max_tries, max_flips, probability))

    max_flips = 9000
    logging.info(f"Starting processing for directory: {directory}")

    # Prepare arguments for processing
    args_list = []
    files = [f for f in os.listdir(directory) if f.endswith('.cnf')]
    for filename in files:
        filepath = os.path.join(directory, filename)
        for mode in modes:
            args_list.append((filename, filepath, mode, max_tries, max_flips, probability))

    max_flips = 10000
    logging.info(f"Starting processing for directory: {directory}")

    # Prepare arguments for processing
    args_list = []
    files = [f for f in os.listdir(directory) if f.endswith('.cnf')]
    for filename in files:
        filepath = os.path.join(directory, filename)
        for mode in modes:
            args_list.append((filename, filepath, mode, max_tries, max_flips, probability))

    max_flips = 11000
    logging.info(f"Starting processing for directory: {directory}")

    # Prepare arguments for processing
    args_list = []
    files = [f for f in os.listdir(directory) if f.endswith('.cnf')]
    for filename in files:
        filepath = os.path.join(directory, filename)
        for mode in modes:
            args_list.append((filename, filepath, mode, max_tries, max_flips, probability))

    max_flips = 12000
    logging.info(f"Starting processing for directory: {directory}")

    # Prepare arguments for processing
    args_list = []
    files = [f for f in os.listdir(directory) if f.endswith('.cnf')]
    for filename in files:
        filepath = os.path.join(directory, filename)
        for mode in modes:
            args_list.append((filename, filepath, mode, max_tries, max_flips, probability))

    max_flips = 13000
    logging.info(f"Starting processing for directory: {directory}")

    # Prepare arguments for processing
    args_list = []
    files = [f for f in os.listdir(directory) if f.endswith('.cnf')]
    for filename in files:
        filepath = os.path.join(directory, filename)
        for mode in modes:
            args_list.append((filename, filepath, mode, max_tries, max_flips, probability))
    
    max_flips = 14000
    logging.info(f"Starting processing for directory: {directory}")

    # Prepare arguments for processing
    args_list = []
    files = [f for f in os.listdir(directory) if f.endswith('.cnf')]
    for filename in files:
        filepath = os.path.join(directory, filename)
        for mode in modes:
            args_list.append((filename, filepath, mode, max_tries, max_flips, probability))

    max_flips = 15000
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
        CREATE TABLE IF NOT EXISTS BenchmarkSubsetIterationsCDF (
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
        logging.info("Table BenchmarkSubsetIterationsCDF created or already exists.")
    
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
                INSERT INTO BenchmarkSubsetIterationsCDF (filename, algorithmname, avg_flips, prob_s, std_flips, tts_99, avg_loops, std_loops, avg_tries, std_tries)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', result)
            conn.commit()
            logging.info("Batch processed and results inserted into the database.")

    conn.close()
    logging.info("Processing complete. Database connection closed. Results in BenchmarkSubsetIterationsCDF.db")

if __name__ == "__main__":
    main()
