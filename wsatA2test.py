import argparse
import time
import random
import networkx as nx
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


# for some reason this is godawfully slow
# First Gruendlage seems to be that performing clause evaluation in 3x x 100 is expensive for larger CNFs
# also I need to add counters to everything

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

def flip_variable(assignment, var):
    assignment[var] = not assignment[var]

def update_unsatisfied_clauses(clauses, assignment, unsatisfied_clauses, flipped_var):
    """
    Update the list of unsatisfied clauses after flipping a variable.
    Only the clauses involving the flipped variable need to be re-evaluated.
    """
    updated_unsat_clauses = unsatisfied_clauses.copy()
    
    for clause in clauses:
        if flipped_var in map(abs, clause):
            clause_eval = evaluate_clause(clause, assignment)
            if clause_eval and clause in updated_unsat_clauses:
                updated_unsat_clauses.remove(clause)
            elif not clause_eval and clause not in updated_unsat_clauses:
                updated_unsat_clauses.append(clause)

    return updated_unsat_clauses

def process_color(color, vars_in_color, unsatisfied_clauses, assignment, p):
    """
    Process a single color to determine which variable to flip.
    Now works with only unsatisfied clauses.
    """
    # Find unsatisfied clauses involving variables of this color
    unsat_clauses_in_color = [
        clause for clause in unsatisfied_clauses
        if any(abs(var) in vars_in_color for var in clause)
    ]

    if not unsat_clauses_in_color:
        return None  # No action needed for this color

    # Randomly select one unsatisfied clause involving this color
    clause = random.choice(unsat_clauses_in_color)

    if random.random() < p:
        # Flip a random variable from the clause that has this color
        vars_in_clause_and_color = [
            abs(var) for var in clause if abs(var) in vars_in_color
        ]
        if not vars_in_clause_and_color:
            return None
        var_to_flip = random.choice(vars_in_clause_and_color)
        return var_to_flip
    else:
        # Flip the variable that minimizes the number of unsatisfied clauses
        break_counts = []
        for var in clause:
            var = abs(var)
            if var in vars_in_color:
                # Temporarily flip the variable
                assignment[var] = not assignment[var]
                num_unsat = len([c for c in unsatisfied_clauses if not evaluate_clause(c, assignment)])
                break_counts.append((num_unsat, var))
                # Undo the flip
                assignment[var] = not assignment[var]
        if not break_counts:
            return None
        min_break = min(break_counts, key=lambda x: x[0])
        vars_with_min_break = [
            var for count, var in break_counts if count == min_break[0]
        ]
        var_to_flip = random.choice(vars_with_min_break)
        return var_to_flip

def ColorWalkSAT(clauses, colors, max_tries, max_loops, p, seed=None):
    if seed is not None:
        random.seed(seed)
    
    variables = list({abs(var) for clause in clauses for var in clause})
    
    # Group variables by color
    color_vars = {}
    for var in variables:
        color = colors.get(var, None)
        if color is not None:
            color_vars.setdefault(color, []).append(var)
        else:
            # Handle variables without a color assignment
            color_vars.setdefault('no_color', []).append(var)
    
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for try_num in range(max_tries):
            # Initialize a random assignment
            assignment = {var: random.choice([True, False]) for var in variables}

            # Initialize the unsatisfied clauses list
            unsatisfied_clauses = get_unsatisfied_clauses(clauses, assignment)

            for loop_num in range(max_loops):
                if not unsatisfied_clauses:
                    return assignment  # Found a satisfying assignment

                vars_to_flip = []

                # Submit all color processing tasks
                futures = {
                    executor.submit(process_color, color, vars_in_color, unsatisfied_clauses, assignment.copy(), p): color
                    for color, vars_in_color in color_vars.items()
                }

                # Collect results as they complete
                for future in as_completed(futures):
                    try:
                        var_to_flip = future.result()
                        if var_to_flip is not None:
                            vars_to_flip.append(var_to_flip)
                    except Exception as e:
                        color = futures[future]
                        print(f"Error processing color {color}: {e}")

                # Apply all collected flips and update unsatisfied clauses
                for var in vars_to_flip:
                    flip_variable(assignment, var)
                    unsatisfied_clauses = update_unsatisfied_clauses(clauses, assignment, unsatisfied_clauses, var)

    return "FAIL"

def main():
    parser = argparse.ArgumentParser(description='ColoringWalksat SAT Solver.')
    parser.add_argument('-cnf', help='Path to SAT problem in .cnf format', required=True)
    parser.add_argument('-p', type=float, help='Probability float between 0 and 1', required=True)
    parser.add_argument('--max_tries', type=int, default=100, help='Maximum number of tries')
    parser.add_argument('--max_loops', type=int, default=1000, help='Maximum number of loops per try')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of parallel workers (defaults to CPU count)')
    args = parser.parse_args()

    filepath = args.cnf
    probability = args.p
    max_tries = args.max_tries
    max_loops = args.max_loops
    num_workers = args.num_workers or multiprocessing.cpu_count()

    # Read and preprocess the CNF file
    try:
        num_vars, clauses = read_dimacs(filepath)
    except Exception as e:
        print(f"Error reading CNF file: {e}")
        return

    # Time how long preprocessing took in wall clock time
    start_color_time = time.perf_counter()

    variables = set(abs(literal) for clause in clauses for literal in clause)

    # Variable adjacency graph
    G = nx.Graph()
    G.add_nodes_from(variables)

    # Edges between variables that appear together in a clause
    for clause in clauses:
        vars_in_clause = set(abs(literal) for literal in clause)
        for var1, var2 in combinations(vars_in_clause, 2):
            G.add_edge(var1, var2)

    # Greedy coloring, returns dictionary of unique colors (0 to n) for each node
    try:
        colors = nx.coloring.greedy_color(G, strategy='largest_first')
    except Exception as e:
        print(f"Error during graph coloring: {e}")
        return

    end_color_time = time.perf_counter()
    time_color = end_color_time - start_color_time
    print(f"Graph coloring completed in {time_color:.4f} seconds.")

    # Running WalkSAT
    start_walksat_time = time.perf_counter()
    result = ColorWalkSAT(clauses, colors, max_tries, max_loops, probability)
    end_walksat_time = time.perf_counter()

    time_walksat = end_walksat_time - start_walksat_time

    if result != "FAIL":
        print("SAT:")
        for var in sorted(result):
            print(f"Variable {var} (Color {colors.get(var, 'N/A')}): {result[var]}")
    else:
        print("No satisfying assignment found within the given limits.")

    print(f"WalkSAT completed in {time_walksat:.4f} seconds.")

if __name__ == "__main__":
    main()
