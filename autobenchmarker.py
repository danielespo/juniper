#!/usr/bin/env python3

import os
import subprocess
import sqlite3
import sys
import multiprocessing
import logging
from multiprocessing import Queue, current_process
from queue import Empty

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("BenchmarkingLog.log"),  # Log file
        # Removed StreamHandler to prevent logging to stdout
    ]
)

def init_database(database_path):
    """Initialize the SQLite database and create the FinalResults table if it doesn't exist."""
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS FinalResults (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                algorithm TEXT,
                p REAL,
                time REAL,
                additional_time REAL,
                max_tries INTEGER,
                max_flips INTEGER,
                tries INTEGER,
                flips INTEGER,
                actual_flips INTEGER
            )
        ''')
        conn.commit()
        conn.close()
        logging.info(f"Database initialized at {database_path}.")
    except sqlite3.Error as e:
        logging.error(f"Failed to initialize database: {e}")
        sys.exit(1)

def run_script(script_path, args):
    """
    Execute a Python script with given arguments and return its output.
    Args:
        script_path (str): Path to the Python script.
        args (list): List of command-line arguments.
    Returns:
        str: Output from the script.
    Raises:
        subprocess.CalledProcessError: If the script execution fails.
    """
    cmd = ['python', script_path] + args
    logging.debug(f"Running command: {' '.join(cmd)}")
    try:
        output = subprocess.check_output(cmd, universal_newlines=True).strip()
        logging.debug(f"Output: {output}")
        return output
    except subprocess.CalledProcessError as e:
        logging.error(f"Command '{' '.join(cmd)}' failed with return code {e.returncode}.")
        raise e

def parse_output(output, expected_parts):
    """
    Parse the output string into components.
    Args:
        output (str): Output string from the script.
        expected_parts (int): Number of expected parts after splitting.
    Returns:
        list: List of parsed elements if successful, None otherwise.
    """
    parts = output.split()
    if len(parts) >= expected_parts:
        return parts
    else:
        logging.warning(f"Unexpected output format: '{output}'")
        return None

def worker(task_queue, result_queue, walksat_path, max_tries, max_flips):
    """
    Worker function to process tasks from the task queue and put results into the result queue.
    Args:
        task_queue (Queue): Queue containing tasks to process.
        result_queue (Queue): Queue to store the results.
        walksat_path (str): Path to the WalkSAT scripts.
        max_tries (int): Maximum number of tries for the algorithms.
        max_flips (int): Maximum number of flips for the algorithms.
    """
    while True:
        try:
            task = task_queue.get_nowait()
        except Empty:
            break  # No more tasks

        cnf_file, cnf_file_path, p = task
        algorithms = {
            'walksat': {'script': 'walksat.py', 'expected_parts': 4, 'additional_arg': '--max_flips'},
            'wsatA2': {'script': 'wsatA2.py', 'expected_parts': 4, 'additional_arg': '--max_loops'},
            'wsatA1': {'script': 'wsatA1.py', 'expected_parts': 4, 'additional_arg': '--max_loops'},
            'wsatB': {'script': 'wsatB.py', 'expected_parts': 4, 'additional_arg': '--max_loops'}
        }

        for algo, details in algorithms.items():
            script_path = os.path.join(walksat_path, details['script'])
            args = [
                '-cnf', cnf_file_path,
                '-p', str(p),
                '--max_tries', str(max_tries),
                details['additional_arg'], str(max_flips)
            ]

            try:
                output = run_script(script_path, args)
                parts = parse_output(output, details['expected_parts'])
                if not parts:
                    logging.error(f"{algo}: Incorrect output format for file {cnf_file} with p={p}.")
                    continue

                # Parse common fields
                time_main = float(parts[0])
                additional_time = float(parts[1]) if algo != 'walksat' else None
                tries = int(parts[2])
                flips = int(parts[3])
                actual_flips = flips  # Adjust if actual_flips is different

                # Prepare the result tuple
                result = (
                    cnf_file,
                    algo,
                    p,
                    time_main,
                    additional_time,
                    max_tries,
                    max_flips,
                    tries,
                    flips,
                    actual_flips
                )

                # Put the result into the result queue
                result_queue.put(result)
                logging.info(f"Processed {algo} for {cnf_file} with p={p}.")

            except subprocess.CalledProcessError:
                logging.error(f"Execution failed for {algo} on {cnf_file} with p={p}.")
            except ValueError as e:
                logging.error(f"Parsing failed for {algo} on {cnf_file} with p={p}: {e}")
            except Exception as e:
                logging.error(f"Unexpected error for {algo} on {cnf_file} with p={p}: {e}")

def database_writer(database_path, result_queue, total_tasks):
    """
    Function to write results from the result queue to the SQLite database.
    Args:
        database_path (str): Path to the SQLite database.
        result_queue (Queue): Queue containing results to write.
        total_tasks (int): Total number of results expected.
    """
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        logging.info("Database writer started.")

        inserted = 0
        while inserted < total_tasks:
            try:
                result = result_queue.get(timeout=5)  # Wait for a result
                cursor.execute('''
                    INSERT INTO FinalResults (
                        filename, algorithm, p, time, additional_time,
                        max_tries, max_flips, tries, flips, actual_flips
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', result)
                inserted += 1
                if inserted % 100 == 0:
                    conn.commit()
                    logging.info(f"Inserted {inserted}/{total_tasks} records.")
            except Empty:
                continue  # Continue waiting for results

        conn.commit()
        conn.close()
        logging.info(f"Database writer completed. Total records inserted: {inserted}.")

    except sqlite3.Error as e:
        logging.error(f"Database writing failed: {e}")

def benchmark(cnf_folder_path, walksat_path, database_path, p_values, max_tries=500, max_flips=50000):
    """
    Main benchmarking function to coordinate the processing of CNF files.
    Args:
        cnf_folder_path (str): Path to the folder containing CNF files.
        walksat_path (str): Path to the WalkSAT scripts.
        database_path (str): Path to the SQLite database.
        p_values (list): List of p values to use.
        max_tries (int): Maximum number of tries for the algorithms.
        max_flips (int): Maximum number of flips for the algorithms.
    """
    # Initialize the database
    init_database(database_path)

    # List all CNF files in the folder
    try:
        cnf_files = [f for f in os.listdir(cnf_folder_path) if f.endswith('.cnf')]
        if not cnf_files:
            logging.error(f"No CNF files found in {cnf_folder_path}.")
            sys.exit(1)
        logging.info(f"Found {len(cnf_files)} CNF files in {cnf_folder_path}.")
    except OSError as e:
        logging.error(f"Error accessing CNF folder: {e}")
        sys.exit(1)

    # Prepare the list of tasks
    tasks = []
    for cnf_file in cnf_files:
        cnf_file_path = os.path.join(cnf_folder_path, cnf_file)
        for p in p_values:
            tasks.append((cnf_file, cnf_file_path, p))

    total_tasks = len(tasks) * 4  # 4 algorithms per task
    logging.info(f"Total benchmarking tasks to process: {total_tasks}.")

    # Create multiprocessing queues
    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    # Populate the task queue
    for task in tasks:
        task_queue.put(task)

    # Start the database writer process
    db_writer = multiprocessing.Process(
        target=database_writer,
        args=(database_path, result_queue, total_tasks)
    )
    db_writer.start()

    # Determine number of processes to use
    num_cpus = os.cpu_count()
    num_processes = max(1, int(num_cpus * 0.6))
    logging.info(f"Using {num_processes} worker processes.")

    # Start worker processes
    workers = []
    for _ in range(num_processes):
        p = multiprocessing.Process(
            target=worker,
            args=(task_queue, result_queue, walksat_path, max_tries, max_flips)
        )
        p.start()
        workers.append(p)
        logging.debug(f"Started worker process PID: {p.pid}")

    # Wait for all workers to finish
    for p in workers:
        p.join()
        logging.debug(f"Worker process PID: {p.pid} has finished.")

    # Wait for the database writer to finish
    db_writer.join()
    logging.info("Benchmarking process completed.")

if __name__ == "__main__":
    # Define paths and parameters
    cnf_folder_path = "/home/dae/SatExperiments/juniper/BenchmarkSubsetPaper"
    walksat_path = "/home/dae/SatExperiments/juniper"
    database_path = "/home/dae/SatExperiments/juniper/FinalResults.db"
    p_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Validate paths
    if not os.path.isdir(cnf_folder_path):
        logging.error(f"The folder {cnf_folder_path} does not exist.")
        sys.exit(1)
    if not os.path.isdir(walksat_path):
        logging.error(f"The folder {walksat_path} does not exist.")
        sys.exit(1)

    # Start benchmarking
    benchmark(cnf_folder_path, walksat_path, database_path, p_values)
