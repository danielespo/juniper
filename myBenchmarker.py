import os
import sqlite3
import myWalksat as mw
import multiprocessing
from multiprocessing import Process, Queue

# Copyright Daniel Espinosa Gonzalez
# October 2024
# UCSB Strukov Lab

def process_algorithm(filename, filepath, mode, max_tries, max_flips, probability, result_queue):
    try:
        avg_flips, prob_s, std_flips, tts_99 = mw.API(
            filepath, max_tries, max_flips, probability, mode)
        result = (filename, mode, avg_flips, prob_s, std_flips, tts_99)
    except Exception as e:
        print(f"    Error processing {filename} with {mode}: {e}")
        result = (filename, mode, None, None, None, None)
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

    # I sprint wrote this rather fast, a better implementation keeps a log of
    # which files finished etc. I will add that functionality soon.
    
    # Old: BenchmarkSubset // now iterations included (extra stats)
    db_file = "BenchmarkSubsetIterations.db"
    modes = ["walksat", "coloringA1_heuristic0", "coloringA1_heuristic1",
             "coloringA1_heuristic2", "coloringA1_heuristic3"]
    max_tries = 10
    max_flips = 100000
    probability = 0.5

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

    batch_size = 4  

    # Process tasks in batches
    for batch_args in batch(args_list, batch_size):
        result_queue = multiprocessing.Queue()
        processes = []

        # Start processes for the current batch
        for args in batch_args:
            filename, filepath, mode, max_tries, max_flips, probability = args
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

    conn.close()
    print("Processing complete. Results stored in BenchmarkSubsetIterations.db.")

if __name__ == "__main__":
    main()
