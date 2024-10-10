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

# After speaking with Dima I realized this needs to change slightly in order to better
# make sense.


# Need to add a candidate list of variables to flip which is fed from the candidate clauses list



# Steps:
# 1) Gather up unsatisfied clauses
# 2) Follow the same algorithm as in WalkSAT to determine cc candidate variables to flip, say set cc_candidates_to_flip
# 3) Random flip a given v, or 
# 4) From candidate clauses with v in C with color k and smallest break-count, pick only variables of v color to flip

def AlgorithmA1(clauses, colors, max_tries, max_loops, p):
    flips = 0
    variables = list(get_variables(clauses))

    for _try in range(max_tries):
        # Initialize a random assignment
        assignment = {var: random.choice([True, False]) for var in variables}

        for _loop in range(max_loops):
            unsat_clauses = get_unsatisfied_clauses(clauses, assignment)
            if not unsat_clauses:
                return assignment, _try, _loop, flips # Success

            cc = unsat_clauses

            # Step 2: Determine cc_candidates_to_flip
            cc_candidates_to_flip = []
            for clause in cc:
                variables_in_clause = [abs(var) for var in clause]
                # Compute break-counts for variables in clause
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

            # Step 3: Choose subset of uncorrelated variables to flip
            # Group candidates by color
            color_to_candidates = {}
            for x, break_count, color in cc_candidates_to_flip:
                color_to_candidates.setdefault(color, []).append((x, break_count))

            # Choose the color with the largest number of variables in cc_candidates_to_flip
            selected_color = max(color_to_candidates.keys(), key=lambda c: len(color_to_candidates[c]))

            candidates_in_color = color_to_candidates[selected_color]

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
