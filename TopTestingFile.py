#!/usr/bin/env python
from multiprocessing import Pool
import logging
import xml.etree.ElementTree as Xet
import logging
import subprocess
import os
import re
import sqlite3

#################################################################################################
# Benchmarking script to populate an existing SQL database with the results of
# a SAT solver (in this case, ColoringWalkSAT) one at a time in parallel using multiple CPUs
# using os to walk through the possibilities of files inside a benchmarking folder
# 
# 
# Usage: Edit the directories which contain your benchmarking instances in this script
# make sure you specify the directory for the compiled SAT solver correctly and with its
# proper syntax ; also rename the .log file before each experiment.
# 
# Daniel A. Espinosa Gonzalez 2024
# Strukov Group, UC Santa Barbara
#
# NOTE: Modified to run python scripts instead to test WALKSAT variations
#       can easily be modded back to run binaries again.
#################################################################################################

logger = logging.getLogger(__name__)

# Change the log file name before using!
logging.basicConfig(filename='ColoringWalksatA2.log', level=logging.INFO)
logger.info('Started')

# Now runs ColoringWalksat, keeping the name to avoid refactor
def run_cadical(cnf_file_path, effort: int):

    cmd = f"python walksatA2.py {cnf_file_path} -{effort}"
    
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    if err:
        logger.info(f"Error executing command for x={cnf_file_path}: {err.decode('utf-8')}")
        logger.info(f"Error executing command {cmd}: {err.decode('utf-8')}")
        return None
    return out.decode('utf-8')

# this had to be fully revamped because the regex is different now
def parse_output_profiling(raw_cmd_data) -> list:
    assert raw_cmd_data is not None, "Error: No data from the command line was passed to parse."
    text = raw_cmd_data
    
    
    if isinstance(text, str):
        matches = pattern_profiling.findall(text)
        parsed_data = []

        for match in matches:
            # Structure each match into a dictionary
            item = {
                "time": float(match[0]),
                "percentage": float(match[1]),
                "name": match[2]
            }
            parsed_data.append(item)
        
        # List of dicts output if success
        return parsed_data
    
    # Nothing by default as a way to stop trash getting into the database
    return []

def insert_data_to_db(data_raw, instance_name):
    # Change to LOCAL database
    # We can also support external SQL databases if we need to do say 29000 SAT problems (as of 2024)
    # that is the number of all the problems in the benchmarking-database.de website ; however using
    # SQLite is much simpler for quick experiments.

    conn = sqlite3.connect('cadical_profiling_benchmarking_data_by_increasing_L_parameter_and_walkeffort.db')
    cursor = conn.cursor()
    
    for item in data_raw:
        L_param = item[0]
        profile_data = item[1]
        walkreleff = item[2]
    
        # Temporary dictionary to store the values
        values = {'instance_name': instance_name, 'L_param': L_param, 'walkreleff': walkreleff}

        # Fill in the values from the profile data, defaulting to None if not present
        profile_keys = ['search', 'stable', 'unstable', 'simplify', 'elim', 'walk', 'probe', 'ternary', 'subsume', 'vivify', 'parse', 'solve']
        for key in profile_keys:
            values[key] = next((item['time'] for item in profile_data if item['name'] == key), None)
        
        # Insert the data into the database using a query
        # SQL queries are not scary, by default make the commands UPPERCASE and the parameters lowercase
        cursor.execute('''
            INSERT INTO cadical_profiling_benchmarking_data_by_increasing_L_parameter_and_walkeffort 
            (instance_name, L_param, walkreleff, search, stable, unstable, simplify, elim, walk, probe, ternary, subsume, vivify, parse, solve)
            VALUES (:instance_name, :L_param, :walkreleff, :search, :stable, :unstable, :simplify, :elim, :walk, :probe, :ternary, :subsume, :vivify, :parse, :solve)
        ''', values)
    
    conn.commit()
    conn.close()

# Change me to change what to do with the scripts
def process_file(args):
    filename, effort = args
    logger.info(f"Starting ColorWSAT with -p{effort} probability for file {filename}")
    try:
        temp = run_cadical(filename, effort)
        if temp is None:
            logger.error(f"Failed to run ColoringWalkSAT for file {filename} with probability {effort}")
            return None
        parsed_temp = parse_output_profiling(temp)
        return [effort, parsed_temp]
    except Exception as e:
        logger.error(f"Error processing file {filename}: {str(e)}")
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Here you can edit all of the different parameters to iterate over for a given solver configuration
    # In this case, I only have probability to test over, keeping the min and max effort as prob

    efforts = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # Change the directory of the benchmarks to consider here !!
    root_dir = "/home/dae/SatExperiments/CryptoEasySBVABenchmarks"
    
    # Create a list of all combinations of filename, effort
    tasks = [
        (os.path.join(dirpath, filename), effort)
        for dirpath, _, filenames in os.walk(root_dir)
        for filename in filenames
        if filename.endswith(".cnf")
        for effort in efforts
    ]

    # Use multiprocessing to parallelize

    n = 32  #Number of CPUs to use
    print('Using', n, 'processes for the pool')

    with Pool(n) as pool: #here was () before
        results = pool.map(process_file, tasks)

    # Group results by filename
    grouped_results = {}
    for (filename, effort), result in zip(tasks, results):
        if result is not None:  # Only process successful results, IMPORTANT !!
            if filename not in grouped_results:
                grouped_results[filename] = []
            grouped_results[filename].append(result)

    # Insert grouped results into the database
    for filename, data in grouped_results.items():
        insert_data_to_db(data, filename)

