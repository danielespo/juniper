import random
import argparse
import networkx as nx
from itertools import combinations
import time
import concurrent.futures  # For parallelization

# Helper functions as provided
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

# New helper function for parallel computation
def compute_break_count(x, clauses, assignment):
    """
    Computes the break count for a single variable.

    Parameters:
    - x (int): The variable to flip.
    - clauses (list): The list of clauses in the CNF.
    - assignment (dict): The current variable assignments.

    Returns:
    - tuple: (variable, break_count)
    """
    local_assignment = assignment.copy()
    # Flip the variable
    local_assignment[x] = not local_assignment[x]
    # Compute the number of clauses that are unsatisfied after the flip
    num_new_unsat = sum(not evaluate_clause(c, local_assignment) for c in clauses)
    return (x, num_new_unsat)

def AlgorithmA2(clauses, colors, max_tries, max_loops, p):
    flips = 0
    variables = list(get_variables(clauses))
    color_set = set(colors.values())

    # Initialize the ProcessPoolExecutor once
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for _try in range(max_tries):
            # Initialize a random assignment
            assignment = {var: random.choice([True, False]) for var in variables}

            for _loop in range(max_loops):
                unsat_clauses = get_unsatisfied_clauses(clauses, assignment)
                if not unsat_clauses:
                    return assignment, _try, _loop, flips  # Success

                # Iterate over each color
                for current_color in color_set:
                    # Step 2: Choose cc clauses from C that contain at least one variable of current_color
                    cc = [
                        clause for clause in unsat_clauses
                        if any(colors[abs(var)] == current_color for var in clause)
                    ]

                    if not cc:
                        continue  # No unsatisfied clauses with this color

                    # Collect all unique variables of current_color from these clauses
                    variables_to_process = set()
                    for clause in cc:
                        vars_in_clause = [abs(var) for var in clause if colors[abs(var)] == current_color]
                        variables_to_process.update(vars_in_clause)

                    if not variables_to_process:
                        continue  # No variables of current_color to process

                    # Submit all break count computations in parallel
                    futures = {
                        executor.submit(compute_break_count, x, clauses, assignment): x
                        for x in variables_to_process
                    }

                    cc_candidates_to_flip = []
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            x, break_count = future.result()
                            cc_candidates_to_flip.append((x, break_count, colors[x]))
                        except Exception as e:
                            print(f"Error computing break count for variable {futures[future]}: {e}")

                    if not cc_candidates_to_flip:
                        continue  # No candidates to flip

                    # Step 3: Since all candidates are of current_color, no need to filter by color again
                    # Select variables to flip from the candidates with minimum break-count
                    min_break_count = min(break_count for _, break_count, _ in cc_candidates_to_flip)
                    vars_with_min_break = [x for x, bc, _ in cc_candidates_to_flip if bc == min_break_count]

                    if not vars_with_min_break:
                        continue  # No variables to flip

                    if random.random() < p:
                        # Random walk move: flip a random variable from the candidates
                        var_to_flip = random.choice(vars_with_min_break)
                    else:
                        # Greedy move: flip the variable with the smallest break-count
                        var_to_flip = random.choice(vars_with_min_break)

                    # Flip the chosen variable
                    flip_variable(assignment, var_to_flip)
                    flips += 1

                    # Break after flipping a variable for the current color to proceed to the next loop iteration
                    break

                else:
                    # If no color led to a flip, continue to the next loop iteration
                    continue

        return "FAIL"

def main():
    parser = argparse.ArgumentParser(description='Algorithm A2 SAT Solver.')
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
    result = AlgorithmA2(clauses, colors, max_tries, max_loops, probability)
    end_colorwalksat_process_time = time.perf_counter()
    time_colorwalksat = end_colorwalksat_process_time - start_colorwalksat_process_time

    if result != "FAIL":
        SAT = 1
        print(time_colorwalksat, time_color, result[1], result[2], result[3])  # Return tries and flips
    else:
        print(0, 0, 0, 0, 0)  # No satisfying assignment found within the given limits

if __name__ == "__main__":
    main()
