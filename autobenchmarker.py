#!/usr/bin/env python3

import os
import subprocess
import sqlite3
import sys
import multiprocessing

# NOTE: I would typically split these steps into
# multiple subscripts, but we are tight on time
# running this WILL strain the server if you are
# benchmarking a LOT of files
# Use with caution!

def process_task(args):
    cnf_file, cnf_file_path, p, walksat_path, max_tries, max_flips, database_path = args

    # Create a connection to the database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Run walksat.py
    walksat_cmd = [
        'python', os.path.join(walksat_path, 'walksat.py'),
        '-cnf', cnf_file_path,
        '-p', str(p),
        '--max_tries', str(max_tries),
        '--max_flips', str(max_flips)
    ]
    try:
        walksat_output = subprocess.check_output(walksat_cmd, universal_newlines=True).strip()
        # Expected output format: time_walksat tries flips
        parts = walksat_output.strip().split()
        if len(parts) >= 3:
            time_walksat = float(parts[0])
            tries = int(parts[1])
            flips = int(parts[2])

            # Insert into database
            cursor.execute('''
                INSERT INTO results (filename, algorithm, p, time, additional_time, max_tries, max_flips, tries, flips)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (cnf_file, 'walksat', p, time_walksat, None, max_tries, max_flips, tries, flips))
            conn.commit()
        else:
            print(f"Unexpected output from walksat.py on {cnf_file} with p={p}: {walksat_output}")
    except subprocess.CalledProcessError as e:
        print(f"Error running walksat.py on {cnf_file} with p={p}: {e}")

    # Run wsatA2.py
    wsatA2_cmd = [
        'python', os.path.join(walksat_path, 'wsatA2.py'),
        '-cnf', cnf_file_path,
        '-p', str(p),
        '--max_tries', str(max_tries),
        '--max_loops', str(max_flips)
    ]
    try:
        wsatA2_output = subprocess.check_output(wsatA2_cmd, universal_newlines=True).strip()
        # Expected output format: time_colorwalksat time_color tries flips
        parts = wsatA2_output.strip().split()
        if len(parts) >= 4:
            time_colorwalksat = float(parts[0])
            time_color = float(parts[1])  # Additional time, e.g., for coloring
            tries = int(parts[2])
            flips = int(parts[3])

            # Insert into database
            cursor.execute('''
                INSERT INTO results (filename, algorithm, p, time, additional_time, max_tries, max_flips, tries, flips)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (cnf_file, 'wsatA2', p, time_colorwalksat, time_color, max_tries, max_flips, tries, flips))
            conn.commit()
        else:
            print(f"Unexpected output from wsatA2.py on {cnf_file} with p={p}: {wsatA2_output}")
    except subprocess.CalledProcessError as e:
        print(f"Error running wsatA2.py on {cnf_file} with p={p}: {e}")

    conn.close()

def benchmark(cnf_folder_path, walksat_path, database_path, p_values):
    # Create or connect to the SQLite3 database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Create the results table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            algorithm TEXT,
            p REAL,
            time REAL,
            additional_time REAL,
            max_tries INTEGER,
            max_flips INTEGER,
            tries INTEGER,
            flips INTEGER
        )
    ''')
    conn.commit()
    conn.close()

    # List all CNF files in the folder
    cnf_files = [f for f in os.listdir(cnf_folder_path) if f.endswith('.cnf')]

    # Shared arguments
    max_tries = 200
    max_flips = 50000  

    # Prepare the list of tasks
    tasks = []
    for cnf_file in cnf_files:
        cnf_file_path = os.path.join(cnf_folder_path, cnf_file)
        for p in p_values:
            tasks.append((cnf_file, cnf_file_path, p, walksat_path, max_tries, max_flips, database_path))

    # Determine number of processes to use
    # Am capping to 60% of CPUs but clearly this is already overwhelming
    # Keeping it to 30% or so.
    num_cpus = os.cpu_count()
    num_processes = max(1, int(num_cpus * 0.6))
    print(f"Using {num_processes} processes.")

    # Use multiprocessing Pool
    with multiprocessing.Pool(processes=num_processes) as pool:
        for _ in pool.imap_unordered(process_task, tasks):
            pass

    print("Benchmarking completed.")

if __name__ == "__main__":

    cnf_folder_path = "/home/dae/SatExperiments/juniper/uf50suiteSATLIB"
    walksat_path = "/home/dae/SatExperiments/juniper"
    database_path = "/home/dae/SatExperiments/juniper/results.db"
    p_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    if not os.path.isdir(cnf_folder_path):
        print(f"Error: The folder {cnf_folder_path} does not exist.")
        sys.exit(1)
    if not os.path.isdir(walksat_path):
        print(f"Error: The folder {walksat_path} does not exist.")
        sys.exit(1)

    benchmark(cnf_folder_path, walksat_path, database_path, p_values)

