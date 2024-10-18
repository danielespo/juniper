import random
import argparse
import networkx as nx
from itertools import combinations
import numpy as np
import time

# python wsatA1.py -cnf problem.cnf -p 0.5 > full_output.txt
# grep "Execution Time" full_output.txt > timing.txt
# vim timing.txt

def read_dimacs(filename):
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

# Algorithm A1 Steps:
# 1) Random assignment
# 2) Gather UNSAT clauses
# 3) From the UNSAT clauses, pick a number say 3 clauses at random. These 3 are the same number of colors.
# 3a) For a clause, either pick a variable at random to pick 
# 3b) Or, from the clause, pick the variable with the least break value
# 4) Gather all the picked variables into a list, this is the candidate list of variables. 
# 5) Heuristics:
# 5a) Flip variables of the color represented with the largest number of variables
# 5b) Randomly
# 5c) Randomly pick variavbles of a color to flip
# 5d) Only pick the first candidate variable
# 6) Additionally, near convergence turn the heuristics off and go back to WalkSAT+SKC.  NOTE: not implemented yet
# 8) END 

def AlgorithmA1(clauses, colors, max_tries, max_loops, p, heuristic_mode=0):
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
    print(unique_colors)
    for _try in range(max_tries):
        # 1) Random assignment
        assignment = np.random.choice([True, False], size=num_vars + 1)  # 1 based indexing
        # Changed it to a vectorized version instead of hash maps to make faster tts

        for _loop in range(max_loops):
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

def main():
    parser = argparse.ArgumentParser(description='Algorithm A1 WSAT Solver.')
    parser.add_argument('-cnf', help='Path to SAT problem in .cnf format', required=True)
    parser.add_argument('-p', type=float, help='Probability float between 0 and 1', required=True)
    parser.add_argument('--max_tries', type=int, default=100, help='Maximum number of tries')
    parser.add_argument('--max_loops', type=int, default=1000, help='Maximum number of loops per try')
    parser.add_argument('--heuristic', type=int, default=0, help='Heuristic mode: 0 = greedy in colors, 1 = random, 2 = random color, 3 = always first cc_var')
    args = parser.parse_args()

    filepath = args.cnf
    probability = args.p
    max_tries = args.max_tries
    max_loops = args.max_loops
    heuristic = args.heuristic
    try:
        num_vars, clauses = read_dimacs(filepath)
    except Exception as e:
        print(f"Error reading CNF file: {e}")
        raise ValueError
    
    start_color_time = time.perf_counter()
    colors = GenerateColors(clauses)
    end_color_time = time.perf_counter()
    time_color = end_color_time - start_color_time

    start_colorwalksat_process_time = time.perf_counter()
    result = AlgorithmA1(clauses, colors, max_tries, max_loops, probability, heuristic)
    end_colorwalksat_process_time = time.perf_counter()
    time_colorwalksat = end_colorwalksat_process_time - start_colorwalksat_process_time

    if result != "FAIL":
        SAT = 1
        # result = [assignment, _try, _loop, flips]
        # NOTE: flips != loop, flips are actual taken flips
        # while loop is iteration count which in walksat is flips
        print(time_colorwalksat, time_color, result[1], result[2], result[3])
    else:
        print(0,0,0,0,0) # No satisfying assignment found within the given limits

if __name__ == "__main__":
    main()

# Original A1 (pure python) before adding numpy to this
# I had to change this because it was unbelievably slow without using numpy
# # - Daniel
# def AlgorithmA1(clauses, colors, max_tries, max_loops, p, heuristic_mode=0):
#     """    
#     clauses: array of clauses from read_cnf() 
#     colors: dictionary of color memberships from GenerateColors()
#     max_tries: integer of max restarts for algorithm
#     max_loops: integer of iterations of algorithm for a given try
#     p: probability of greedy vs random selection
#     heuristic_mode: 
#         0 = greedy in colors from candidate variables to flip
#         1 = random from candidate variables to flip
#         2 = pick a random color from candidate variables to flip
#         3 = always pick first candidate variable in candidate variables to flip
#     """

#     flips = 0
#     variables = np.array(sorted(get_variables(clauses)))
    
#     for _try in range(max_tries):
#         # 1) Random assignment
#         assignment = {var: random.choice([True, False]) for var in variables}

#         for _loop in range(max_loops):
#             # 2) Gather UNSAT clauses
#             unsat_clauses = get_unsatisfied_clauses(clauses, assignment)
#             if not unsat_clauses:
#                 return assignment, _try, _loop, flips # Success

#             # 3) From the UNSAT clauses, pick candidate clauses up to n_number_colors 
#             random_samples_count = len(colors.keys())

#             # Ensure we don't sample more clauses than there are UNSAT clauses
#             # this was the previous bug.
#             random_samples_count = min(random_samples_count, len(unsat_clauses))

#             selected_clauses = random.sample(unsat_clauses, random_samples_count)
#             cc_candidates_to_flip = []
#             for clause in selected_clauses:
#                 variables_in_clause = [abs(var) for var in clause]

#                 # Pick a random variable of the clause for a candidate variable 
#                 if random.random() < p:
#                     x = random.choice(variables_in_clause)
#                     cc_candidates_to_flip.append((x, colors[x]))

#                 # Or pick variable with least break-count
#                 else:
#                     break_count = []
#                     for x in variables_in_clause:
#                         # Break-count is the number of clauses that become unsatisfied when flipping x
#                         flip_variable(assignment, x)
#                         num_new_unsat = sum(
#                             not evaluate_clause(c, assignment) for c in clauses if evaluate_clause(c, assignment)
#                         )
#                         flip_variable(assignment, x)
#                         break_count.append(num_new_unsat)
                    
#                     # Find indices with least break count
#                     min_break = min(break_count)
#                     indices = [i for i, count in enumerate(break_count) if count == min_break]

#                     # Pick one index at random if tied, else always picks least
#                     idx = random.choice(indices)
#                     x = variables_in_clause[idx]
#                     cc_candidates_to_flip.append((x, colors[x]))


#             # 4) Gather all the picked variables into the candidate list of variables. 
#             color_to_candidates = {}
#             for x, color in cc_candidates_to_flip:
#                 color_to_candidates.setdefault(color, []).append((x))


#             # 5) Heuristically pick variables to flip. 
#             # 5a) Flip variables of the color represented with the largest number of variables
#             if heuristic_mode == 0:
#                 selected_color = max(color_to_candidates.keys(), key=lambda c: len(color_to_candidates[c]))
#                 candidates_in_color = color_to_candidates[selected_color]

#                 # this was much more expensive in this implementation than in the numpy one
#                 for i in candidates_in_color:
#                     flip_variable(assignment, i)
#                     flips += 1

#             # 5b) Randomly pick a variable from candidate variables to flip and flip it
#             elif heuristic_mode == 1:
#                 var_to_flip = random.choice(cc_candidates_to_flip)[0]  # Extract variable x
#                 flip_variable(assignment, var_to_flip)
#                 flips += 1

#             # 5c) Randomly pick a color, flip all variables of that color
#             elif heuristic_mode == 2:
#                 selected_color = random.choice(list(color_to_candidates.keys()))
#                 candidates_in_color = color_to_candidates[selected_color]

#                 for i in candidates_in_color:
#                     flip_variable(assignment, i)
#                     flips += 1


#             # 5d) Always pick the first variable in candidate variables to flip and flip it
#             elif heuristic_mode == 3:
#                 var_to_flip = cc_candidates_to_flip[0][0]  # Extract variable x
#                 flip_variable(assignment, var_to_flip)
#                 flips += 1

#     return "FAIL"