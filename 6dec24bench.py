#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import random
import argparse
import networkx as nx
from itertools import combinations
import time


# In[2]:


def read_dimacs(filename): # int num_vars, array clauses
    clauses = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('p'):
                parts = line.strip().split()
                if len(parts) >= 4:
                    _, _, variables, _ = parts[:4]
                    num_vars = int(variables)
                else:
                    raise ValueError("Invalid CNF file format in 'p' line.")
            elif line.startswith('c') or line.startswith('%') or line.startswith('0'):
                continue
            else:
                clause = list(map(int, line.strip().split()))
                # Remove the trailing 0 if present
                if clause and clause[-1] == 0:
                    clause = clause[:-1]
                if clause:
                    clauses.append(clause)
    return num_vars, clauses


# In[3]:


def evaluate_clause(clause, assignment):
    return any(
        (var > 0 and assignment.get(abs(var), False)) or
        (var < 0 and not assignment.get(abs(var), False))
        for var in clause
    )

def get_unsatisfied_clauses(clauses, assignment):
    return [clause for clause in clauses if not evaluate_clause(clause, assignment)]

def get_variables(clauses):
    return set(abs(var) for clause in clauses for var in clause)

def flip_variable(assignment, var):
    assignment[var] = not assignment[var]

def GenerateColors(clauses):
    variables = set(abs(literal) for clause in clauses for literal in clause)
    G = nx.Graph()
    G.add_nodes_from(variables)  # Variable adjacency graph

    # If variables appear in the same clause, make an edge
    for clause in clauses:
        vars_in_clause = set(abs(literal) for literal in clause)
        for var1, var2 in combinations(vars_in_clause, 2):
            G.add_edge(var1, var2)

    # Greedy coloring , return dictionary of unique colors (0 to n) for each node
    colors = nx.coloring.greedy_color(G, strategy='largest_first')
    return colors

def AlgorithmA1(clauses, colors, max_tries, max_flips, p, heuristic_mode=0):
    """    
    clauses: array of clauses from read_cnf() 
    colors: dictionary of color memberships from GenerateColors()
    max_tries: integer of max restarts for algorithm
    max_loops: integer of iterations of algorithm for a given try
    p: probability of greedy vs random selection
    heuristic_mode: 
        0 = greedy in colors from candidate variables to flip
        1 = random from candidate variables to flip
        2 = pick a random color from candidate variables to flip
        3 = always pick first candidate variable in candidate variables to flip
    """
    flips = 0
    variables = np.array(sorted(get_variables(clauses))) # Sorted list of variables
    # 1-based indexing, because 0 in cnf files is used for line breaks
    num_vars = variables[-1]
    color_array = np.zeros(num_vars + 1, dtype=int)
    for var, color in colors.items():
        color_array[var] = color

    # Get the number of unique colors
    unique_colors = np.unique(color_array[color_array > 0])
    
    for _try in range(max_tries):
        # 1) Random assignment
        assignment = np.random.choice([True, False], size=num_vars + 1)  # 1 based indexing
        # Changed it to a vectorized version instead of hash maps to make faster tts

        # 11/11/2024 2:32pm
        # Loops != Tries
        # In regular walksat, we report tries, the general idea here was to
        # increase the parallelism of the algorithm by splitting uncorrelated flips
        # but by adding this "loops" parameter, I underreport the number of tries
        # since there are many loops that occur under a given 'try'.

        for _loop in range(max_flips):
            # 2) Gather UNSAT clauses
            unsat_indices = []
            for idx, clause in enumerate(clauses):
                literals = np.array(clause, dtype=int)
                vars_in_clause = np.abs(literals).astype(int)
                signs = literals > 0
                clause_values = assignment[vars_in_clause]  # Get assignment values
                clause_evaluation = np.where(signs, clause_values, ~clause_values)
                # If none evaluate to 1, add to unsat index
                if not np.any(clause_evaluation):
                    unsat_indices.append(idx)

            if not unsat_indices:
                return assignment[1:], _try, _loop, flips  # Success, return

            # 3) From the UNSAT clauses, pick candidate clauses based on how many colors
            random_samples_count = min(len(unique_colors), len(unsat_indices))
            # Ensure we don't sample more clauses than there are UNSAT clauses
            # this was a bug in the other implementation as well.
            selected_indices = np.random.choice(unsat_indices, size=random_samples_count, replace=False)
            selected_clauses = [clauses[i] for i in selected_indices]

            cc_candidates_to_flip = []
            for clause in selected_clauses:
                variables_in_clause = np.abs(np.array(clause, dtype=int))
                
                # Pick a random variable of the clause for a candidate variable 
                if random.random() < p:
                    # Randomly pick a variable from the clause
                    x = np.random.choice(variables_in_clause)
                    cc_candidates_to_flip.append((x, color_array[x]))
                
                # Or pick variable with least break-count
                else:
                    break_counts = []
                    # Break-count is the number of clauses that become unsatisfied when flipping x
                    for x in variables_in_clause:
                        # Flip variable
                        assignment[x] = ~assignment[x]
                        # Evaluate the number of clauses that become unsatisfied
                        num_new_unsat = 0
                        # Same as before...
                        for clause_check in clauses: 
                            literals_check = np.array(clause_check, dtype=int)
                            vars_in_clause_check = np.abs(literals_check).astype(int)
                            signs_check = literals_check > 0
                            clause_values_check = assignment[vars_in_clause_check]
                            clause_evaluation_check = np.where(signs_check, clause_values_check, ~clause_values_check)
                            if not np.any(clause_evaluation_check):
                                num_new_unsat += 1

                        break_counts.append(num_new_unsat)
                        # Flip variable back
                        assignment[x] = ~assignment[x]

                    # Find variables with least break count
                    min_break = np.min(break_counts)
                    min_indices = np.where(break_counts == min_break)[0]
                    
                    # Pick one index at random if there is more than one this is the tiebreaker
                    idx_min = np.random.choice(min_indices)
                    x = variables_in_clause[idx_min]
                    cc_candidates_to_flip.append((x, color_array[x]))

            # 4) Gather all the picked variables into the candidate list of variables
            color_to_candidates = {}
            for x, color in cc_candidates_to_flip:
                color_to_candidates.setdefault(color, []).append(x)

            # 5) Heuristically pick which variables to flip:
            if heuristic_mode == 0:
                # 5a) Flip variables of the color with the largest number of variables
                selected_color = max(color_to_candidates.keys(), key=lambda c: len(color_to_candidates[c]))
                candidates_in_color = color_to_candidates[selected_color]
                assignment[candidates_in_color] = ~assignment[candidates_in_color]
                flips += len(candidates_in_color)
            elif heuristic_mode == 1:
                # 5b) Randomly pick a variable from candidate variables to flip
                var_to_flip = np.random.choice([x for x, _ in cc_candidates_to_flip])
                assignment[var_to_flip] = ~assignment[var_to_flip]
                flips += 1
            elif heuristic_mode == 2:
                # 5c) Randomly pick a color, flip all variables of that color
                selected_color = np.random.choice(list(color_to_candidates.keys()))
                candidates_in_color = color_to_candidates[selected_color]
                assignment[candidates_in_color] = ~assignment[candidates_in_color]
                flips += len(candidates_in_color)
            elif heuristic_mode == 3:
                # 5d) Always pick the first candidate variable to flip
                var_to_flip = cc_candidates_to_flip[0][0]
                assignment[var_to_flip] = ~assignment[var_to_flip]
                flips += 1

    return "FAIL"


# In[4]:


def walkSAT(clauses, max_tries, max_flips, p): # assignment, _Tries, _Flips, flips
    flips = 0
    def evaluate_clause(clause, assignment):
        return any((var > 0 and assignment.get(abs(var), False)) or 
                   (var < 0 and not assignment.get(abs(var), False)) for var in clause)

    def get_unsatisfied_clauses(clauses, assignment):
        return [clause for clause in clauses if not evaluate_clause(clause, assignment)]

    def get_variables(clauses):
        return set(abs(var) for clause in clauses for var in clause)

    def flip_variable(assignment, var):
        assignment[var] = not assignment[var]

    for _Tries in range(max_tries):
        variables = list(get_variables(clauses))
        assignment = {var: random.choice([True, False]) for var in variables}
        
        for _Flips in range(max_flips):

            unsatisfied = get_unsatisfied_clauses(clauses, assignment)
            if not unsatisfied:
                return assignment, _Tries, _Flips, flips  # Found a satisfying assignment
            
            clause = random.choice(unsatisfied)
            if random.random() < p:
                # Flip a random variable from the clause
                var_to_flip = abs(random.choice(clause))
            else:
                # Flip a variable that minimizes the number of unsatisfied clauses if flipped
                break_counts = []
                for var in clause:
                    assignment[abs(var)] = not assignment[abs(var)]
                    break_counts.append((len(get_unsatisfied_clauses(clauses, assignment)), abs(var)))
                    assignment[abs(var)] = not assignment[abs(var)]  # Undo the flip
                
                min_break = min(break_counts, key=lambda x: x[0])
                vars_with_min_break = [var for break_count, var in break_counts if break_count == min_break[0]]
                var_to_flip = random.choice(vars_with_min_break)
            
            flip_variable(assignment, var_to_flip)
            flips += 1

    return "FAIL"


# In[ ]:


# Part of the code that runs the algorithm in parallel
import os
import glob
import time
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import glob
import time
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import json

def extract_problem_number(filename) -> int:
    """Extract numerical problem number from filename."""
    basename = os.path.basename(filename)
    num_str = ''.join(filter(str.isdigit, basename))
    return int(num_str)

def process_file(cnf_file):
    import time

    # Same as nature comm paper
    MAX_FLIPS = 10000
    MAX_TRIES = 200

    problem_name = os.path.basename(cnf_file)
    num_vars, clauses = read_dimacs(cnf_file)
    colors = GenerateColors(clauses)

    successes = 0
    flips_success = []

    for _ in range(MAX_TRIES):
        start = time.perf_counter()
        # Which A1 to pick: that which minimizes wasted computations
        solution_object = AlgorithmA1(clauses, colors, max_tries=1, max_flips=MAX_FLIPS, p=0.5, heuristic_mode=0)
        end = time.perf_counter()

        if solution_object != "FAIL":
            successes += 1
            flips_success.append(solution_object[2])

    success_rate = successes / MAX_TRIES
    p_hat = success_rate
    if successes > 0:
        avg_flips = np.mean(flips_success)
    else:
        avg_flips = np.nan

    if success_rate > 0:
        MR = avg_flips + ((1 - success_rate) / success_rate * MAX_FLIPS)
    else:
        MR = float('inf')

    print(f"{problem_name}: Successes = {successes}/{MAX_TRIES}, Success Rate = {success_rate:.4f}, "
          f"Avg. Flips = {avg_flips}, MR = {MR}, p_hat = {p_hat:.4f}")

    # Return these results so we can store them and plot later
    return {
        "problem_name": problem_name,
        "successes": successes,
        "max_tries": MAX_TRIES,
        "success_rate": success_rate,
        "p_hat": p_hat,
        "avg_flips": avg_flips,
        "MR": MR
    }

def main():
    cnf_directory = "/home/dae/SatExperiments/juniper/uf50suiteSATLIB/"
    cnf_files = glob.glob(os.path.join(cnf_directory, "uf50*.cnf"))
    cnf_files.sort(key=extract_problem_number)

    total_cpus = multiprocessing.cpu_count()
    num_workers = max(1, int(total_cpus * 0.2))
    print(f"Total CPU cores: {total_cpus}, Using {num_workers} worker processes.")

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(process_file, cnf_files)

    if not results:
        print("No results.")
        return

    # Convert results to arrays for plotting
    # Take only first 100 for plots
    results_for_plot = results[:100]

    problem_names = [r["problem_name"] for r in results_for_plot]
    p_hats = np.array([r["p_hat"] for r in results_for_plot], dtype=float)
    MRs = np.array([r["MR"] for r in results_for_plot], dtype=float)

    # Plot p_hat
    plt.figure(figsize=(14, 7))
    plt.plot(problem_names, p_hats, marker='o', linestyle='-', label='p_hat (Success Probability)')
    plt.xlabel('Problem Name')
    plt.ylabel('p_hat')
    plt.title('Estimated p_hat for First 100 Problems')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot MR
    plt.figure(figsize=(14, 7))
    plt.plot(problem_names, MRs, marker='o', linestyle='-', label='MR (Mean Run-Length)')
    plt.xlabel('Problem Name')
    plt.ylabel('MR')
    plt.title('Mean Run-Length (MR) for First 100 Problems')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Store results in JSON for later inspection
    # The file will contain all results, not just the first 100
    output_file = "results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results stored in {output_file}")

if __name__ == "__main__":
    main()

