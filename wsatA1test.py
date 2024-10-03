import random
import argparse
import networkx as nx
from itertools import combinations

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

def AlgorithmA1(clauses, colors, max_tries, max_loops, p):
    variables = list(get_variables(clauses))

    color_vars = {}
    for var in variables:
        color = colors[var]
        color_vars.setdefault(color, []).append(var)

    for _try in range(max_tries):
        # Initialize a random assignment
        assignment = {var: random.choice([True, False]) for var in variables}

        for _loop in range(max_loops):
            unsat_clauses = get_unsatisfied_clauses(clauses, assignment)
            if not unsat_clauses:
                return assignment  # Success

            # Step 1: Choose cc UNSAT clauses (here we use all unsatisfied clauses)
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

            # Choose a color based on desired strategy
            # For example, choose the color with the largest number of variables in cc_candidates_to_flip
            selected_color = max(color_to_candidates.keys(), key=lambda c: len(color_to_candidates[c]))

            candidates_in_color = color_to_candidates[selected_color]

            # Select variables to flip from the chosen color
            # Let's pick variables with minimum break-count
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

    return "FAIL"

def main():
    parser = argparse.ArgumentParser(description='Algorithm A1 SAT Solver.')
    parser.add_argument('-cnf', help='Path to SAT problem in .cnf format', required=True)
    parser.add_argument('-p', type=float, help='Probability float between 0 and 1', required=True)
    parser.add_argument('--max_tries', type=int, default=100, help='Maximum number of tries')
    parser.add_argument('--max_loops', type=int, default=1000, help='Maximum number of loops per try')
    args = parser.parse_args()

    filepath = args.cnf
    probability = args.p
    max_tries = args.max_tries
    max_loops = args.max_loops

    # Read the CNF file
    try:
        num_vars, clauses = read_dimacs(filepath)
    except Exception as e:
        print(f"Error reading CNF file: {e}")
        raise ValueError

    # Generate colors using the provided GenerateColors function
    colors = GenerateColors(clauses)

    # Run Algorithm A1
    result = AlgorithmA1(clauses, colors, max_tries, max_loops, probability)

    if result != "FAIL":
        print("Satisfying assignment found:")
        for var in sorted(result.keys()):
            val = assignment_value = 1 if result[var] else 0
            print(f"Variable {var} = {assignment_value}")
    else:
        print("No satisfying assignment found within the given limits.")

if __name__ == "__main__":
    main()
