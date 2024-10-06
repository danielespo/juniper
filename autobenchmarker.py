#!/usr/bin/env python3

import os
import subprocess
import sqlite3
import sys
import multiprocessing

# Oct 10 - Added A1, A2, B, and compared to wsat

def process_task(args):
    cnf_file, cnf_file_path, p, walksat_path, max_tries, max_flips, database_path, lock = args

    # Create a connection to the database within a lock to prevent concurrency issues
    with lock:
        try:
            conn = sqlite3.connect(database_path)
            cursor = conn.cursor()
        except sqlite3.Error as e:
            print(f"Database connection error for {cnf_file} with p={p}: {e}")
            return

    # Define a helper function to run a solver
    def run_solver(solver_name, script, cmd_args, expected_parts, additional_time_required):
        solver_cmd = [sys.executable, os.path.join(walksat_path, script)] + cmd_args
        try:
            output = subprocess.check_output(solver_cmd, universal_newlines=True).strip()
            parts = output.split()

            # Handle UNSAT cases where output is '0 0 0 0 0'
            if parts == ['0', '0', '0', '0', '0']:
                time_main = 0.0
                time_color = 0.0 if additional_time_required else None
                tries = 0
                flips = 0
                actual_flips = 0
            else:
                if len(parts) < expected_parts:
                    print(f"Unexpected output from {script} on {cnf_file} with p={p}: {output}")
                    return

                time_main = float(parts[0])
                if additional_time_required:
                    time_color = float(parts[1])
                    tries = int(parts[2])
                    flips = int(parts[3])
                    actual_flips = int(parts[4])
                else:
                    time_color = None
                    tries = int(parts[1])
                    flips = int(parts[2])
                    actual_flips = int(parts[3])

            # Insert into database
            with lock:
                try:
                    cursor.execute('''
                        INSERT INTO FinalResults (
                            filename, algorithm, p, time, additional_time,
                            max_tries, max_flips, tries, flips, actual_flips
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        cnf_file,
                        solver_name,
                        p,
                        time_main,
                        time_color,
                        max_tries,
                        max_flips,
                        tries,
                        flips,
                        actual_flips
                    ))
                    conn.commit()
                except sqlite3.Error as e:
                    print(f"Database insertion error for {solver_name} on {cnf_file} with p={p}: {e}")

        except subprocess.CalledProcessError as e:
            print(f"Error running {script} on {cnf_file} with p={p}: {e}")
        except ValueError as ve:
            print(f"Value parsing error from {script} on {cnf_file} with p={p}: {ve}")
        except Exception as ex:
            print(f"Unexpected error with {script} on {cnf_file} with p={p}: {ex}")

    # Run walksat.py
    run_solver(
        solver_name='walksat',
        script='walksat.py',
        cmd_args=[
            '-cnf', cnf_file_path,
            '-p', str(p),
            '--max_tries', str(max_tries),
            '--max_flips', str(max_flips)
        ],
        expected_parts=4,
        additional_time_required=False
    )

    # Run wsatA2.py
    run_solver(
        solver_name='wsatA2',
        script='wsatA2.py',
        cmd_args=[
            '-cnf', cnf_file_path,
            '-p', str(p),
            '--max_tries', str(max_tries),
            '--max_loops', str(max_flips)
        ],
        expected_parts=5,
        additional_time_required=True
    )

    # Run wsatA1.py
    run_solver(
        solver_name='wsatA1',
        script='wsatA1.py',
        cmd_args=[
            '-cnf', cnf_file_path,
            '-p', str(p),
            '--max_tries', str(max_tries),
            '--max_loops', str(max_flips)
        ],
        expected_parts=5,
        additional_time_required=True
    )

    # Run wsatB.py
    run_solver(
        solver_name='wsatB',
        script='wsatB.py',
        cmd_args=[
            '-cnf', cnf_file_path,
            '-p', str(p),
            '--max_tries', str(max_tries),
            '--max_loops', str(max_flips)
        ],
        expected_parts=5,
        additional_time_required=True
    )

    # Close the database connection
    with lock:
        conn.close()

def benchmark(cnf_folder_path, walksat_path, database_path, p_values):
    # Initialize a multiprocessing lock
    lock = multiprocessing.Lock()

    # Create or connect to the SQLite3 database
    with lock:
        try:
            conn = sqlite3.connect(database_path)
            cursor = conn.cursor()

            # Create the FinalResults table if it doesn't exist
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
        except sqlite3.Error as e:
            print(f"Database initialization error: {e}")
            sys.exit(1)
        finally:
            conn.close()

    # List all CNF files in the folder
    try:
        cnf_files = [f for f in os.listdir(cnf_folder_path) if f.endswith('.cnf')]
        if not cnf_files:
            print(f"No CNF files found in {cnf_folder_path}.")
            return
    except Exception as e:
        print(f"Error accessing CNF folder {cnf_folder_path}: {e}")
        sys.exit(1)

    # Shared arguments
    max_tries = 500
    max_flips = 50000  

    # Prepare the list of tasks
    tasks = []
    for cnf_file in cnf_files:
        cnf_file_path = os.path.join(cnf_folder_path, cnf_file)
        for p in p_values:
            tasks.append((cnf_file, cnf_file_path, p, walksat_path, max_tries, max_flips, database_path, lock))

    # Determine number of processes to use
    # Capping to 60% of CPUs :)
    num_cpus = os.cpu_count()
    num_processes = max(1, int(num_cpus * 0.6))
    print(f"Using {num_processes} processes.")

    # Use multiprocessing Pool
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_task, tasks)

    print("Benchmarking completed.")

if __name__ == "__main__":

    cnf_folder_path = "/home/dae/SatExperiments/juniper/TestFolderCNF"
    walksat_path = "/home/dae/SatExperiments/juniper"
    database_path = "/home/dae/SatExperiments/juniper/FinalResults.db"
    p_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    if not os.path.isdir(cnf_folder_path):
        print(f"Error: The folder {cnf_folder_path} does not exist.")
        sys.exit(1)
    if not os.path.isdir(walksat_path):
        print(f"Error: The folder {walksat_path} does not exist.")
        sys.exit(1)

    benchmark(cnf_folder_path, walksat_path, database_path, p_values)
