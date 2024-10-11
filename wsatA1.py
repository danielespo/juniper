import random
import argparse
import networkx as nx
from itertools import combinations
import time

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

    # Greedy coloring , returns dictionary of unique colors (0 to n) for each node
    colors = nx.coloring.greedy_color(G, strategy='largest_first')
    return colors


# Update Oct 10 2024
# After speaking with Dima I realized this needs to change
# Adding a candidate list of variables to flip which is fed 
# from the candidate clauses list

# Steps:

# 1) Random assignment
# 2) Gather UNSAT clauses
# 3) From the UNSAT clauses, pick a number say 3 clauses at random. These 3 are the same number of colors.
# 3a) For a clause, either pick a variable at random to pick 
# 3b) Or, from the clause, pick the variable with the least break value
# 4) Gather all the picked variables into a list, this is the candidate list of variables. 
# 5) Now, heuristically, you can pick variables from the same color and flip them because they are uncorrelated 
# 5a) Flip variables of the color represented with the largest number of variables
# 5b) Randomly
# 5c) Randomly pick variavbles of a color to flip
# 6) Additionally, near convergence turn the heuristics off and go back to WalkSAT/SKC. 
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
        4 = always pick last candidate variable in candidate variables to flip
    """

    flips = 0
    variables = list(get_variables(clauses))

    for _try in range(max_tries):
        # 1) Random assignment
        assignment = {var: random.choice([True, False]) for var in variables}

        for _loop in range(max_loops):
            # 2) Gather UNSAT clauses
            unsat_clauses = get_unsatisfied_clauses(clauses, assignment)
            if not unsat_clauses:
                return assignment, _try, _loop, flips # Success

            random_samples_count = len(colors) #should be int
            cc = random.sample(unsat_clauses, int(random_samples_count)) # random list as long as colors

            # 3) From the UNSAT clauses, pick a number, say 3 clauses at random. 
            # Where 3 here is the number of colors.
            cc_candidates_to_flip = []
            for clause in cc:
                variables_in_clause = [abs(var) for var in clause]

                # Pick either a random variable of the clause or 
                if random.random() < p:
                    temp_var = random.choice(variables_in_clause)
                    cc_candidates_to_flip.append(temp_var)

                # Compute break-counts for variables in clause
                else:
                    for x in variables_in_clause:
                        # Compute break-count for x
                        # Break-count is the number of clauses that become unsatisfied when flipping x
                        # Since we don't have variable-to-clauses mapping, we can approximate it
                        flip_variable(assignment, x)
                        num_new_unsat = sum(
                            not evaluate_clause(c, assignment) for c in clauses if evaluate_clause(c, assignment)
                        )
                        flip_variable(assignment, x)  # Flip back
                        break_count = num_new_unsat
                        cc_candidates_to_flip.append((x, break_count, colors[x]))

            # 4) Gather all the picked variables into a list, this is the candidate list of variables. 
            # This is what cc_candidates_to_flip is.

            # Group candidates by color
            color_to_candidates = {} # Now a dict with vars, color of var; this could've been precomputed
            for x, break_count, color in cc_candidates_to_flip:
                color_to_candidates.setdefault(color, []).append((x, break_count))

            # Choose the color with the largest number of variables in cc_candidates_to_flip
            selected_color = max(color_to_candidates.keys(), key=lambda c: len(color_to_candidates[c]))

            candidates_in_color = color_to_candidates[selected_color]

            # Now, we can just flip the variables of that color

            # OR we can flip randomly from the list

            # OR we can pick a random different color to flip from

            # Select variables to flip from the chosen color with minimum break-count
            min_break_count = min(break_count for x, break_count in candidates_in_color)
            vars_with_min_break = [x for x, bc in candidates_in_color if bc == min_break_count]

            if random.random() < p:
                # Random walk move: flip a random variable from the candidates
                var_to_flip = random.choice(vars_with_min_break)
            else:
                # Greedy move: flip the variable with the smallest break-count
                var_to_flip = random.choice(vars_with_min_break)

            # Flip the chosen variable
            flip_variable(assignment, var_to_flip)
            flips += 1

    return "FAIL"

def main():
    parser = argparse.ArgumentParser(description='Algorithm A1 WSAT Solver.')
    parser.add_argument('-cnf', help='Path to SAT problem in .cnf format', required=True)
    parser.add_argument('-p', type=float, help='Probability float between 0 and 1', required=True)
    parser.add_argument('--max_tries', type=int, default=100, help='Maximum number of tries')
    parser.add_argument('--max_loops', type=int, default=1000, help='Maximum number of loops per try')
    args = parser.parse_args()

    filepath = args.cnf
    probability = args.p
    max_tries = args.max_tries
    max_loops = args.max_loops

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
    result = AlgorithmA1(clauses, colors, max_tries, max_loops, probability)
    end_colorwalksat_process_time = time.perf_counter()
    time_colorwalksat = end_colorwalksat_process_time - start_colorwalksat_process_time

    if result != "FAIL":
        SAT = 1
        print(time_colorwalksat, time_color, result[1], result[2], result[3]) # Return tries and flips
    else:
        print(0,0,0,0,0) # No satisfying assignment found within the given limits

if __name__ == "__main__":
    main()
