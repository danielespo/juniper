import random
import argparse
import networkx as nx
from itertools import combinations

def read_dimacs(filename):
    clauses = []
    num_vars = 0
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '' or line[0] in ('c', '%', '0'):
                continue
            if line[0] == 'p':
                parts = line.split()
                num_vars = int(parts[2])
            else:
                clause = list(map(int, line.strip().split()))
                # Remove trailing zero if present
                if clause and clause[-1] == 0:
                    clause = clause[:-1]
                if clause:
                    clauses.append(clause)
    return num_vars, clauses

# Function to evaluate if a clause is satisfied under the current assignment
def evaluate_clause(clause, assignment):
    for var in clause:
        x = abs(var)
        val = assignment.get(x, False)
        if var > 0 and val:
            return True
        elif var < 0 and not val:
            return True
    return False

# Function to get all unsatisfied clauses under the current assignment
def get_unsatisfied_clauses(clauses, assignment):
    unsat_clauses = []
    for idx, clause in enumerate(clauses):
        if not evaluate_clause(clause, assignment):
            unsat_clauses.append((idx, clause))
    return unsat_clauses

# Function to get all variables present in the clauses
def get_variables(clauses):
    variables = set()
    for clause in clauses:
        for var in clause:
            variables.add(abs(var))
    return variables

# Function to flip the value of a variable in the assignment
def flip_variable(assignment, var):
    assignment[var] = not assignment[var]

# Function to build the graph G of literals
def build_graph(clauses):
    variables = get_variables(clauses)
    G = nx.Graph()
    G.add_nodes_from(variables)
    for clause in clauses:
        vars_in_clause = set(abs(var) for var in clause)
        for var1, var2 in combinations(vars_in_clause, 2):
            G.add_edge(var1, var2)
    return G

# Function to compute the break-count of a variable
def compute_break_count(x, clauses, assignment, variable_to_clauses):
    break_count = 0
    for idx in variable_to_clauses[x]:
        clause = clauses[idx]
        if evaluate_clause(clause, assignment):
            # Flip x temporarily
            assignment[x] = not assignment[x]
            clause_satisfied_after_flip = evaluate_clause(clause, assignment)
            # Flip x back
            assignment[x] = not assignment[x]
            if not clause_satisfied_after_flip:
                break_count += 1
    return break_count

# Main Algorithm A1 implementation
def AlgorithmA1(clauses, colors, variable_to_clauses, max_tries, max_loops, p):
    variables = list(get_variables(clauses))
    c = set(colors.values())
    color_vars = {}
    for var in variables:
        color = colors[var]
        color_vars.setdefault(color, []).append(var)

    for i in range(max_tries):
        # Initialize a random assignment
        assignment = {var: random.choice([True, False]) for var in variables}

        for j in range(max_loops):
            unsat_clauses = get_unsatisfied_clauses(clauses, assignment)
            if not unsat_clauses:
                return assignment  # Success

            # Get unsatisfied clauses
            cc = unsat_clauses

            # For each color
            for v in c:
                vars_in_color = color_vars[v]
                # Clauses with at least one variable of the current color v
                f_indices = set()
                for var in vars_in_color:
                    f_indices.update(variable_to_clauses[var])

                # Unsatisfied clauses in f
                unsat_clauses_in_f = [cc_item for cc_item in cc if cc_item[0] in f_indices]

                if not unsat_clauses_in_f:
                    continue  # No unsatisfied clauses involving this color

                # Choose an unsatisfied clause at random
                idx_C, C = random.choice(unsat_clauses_in_f)

                # Variables in the clause
                variables_in_C = [abs(var) for var in C]

                # Compute break-counts for variables in C
                break_counts = {}
                for x in variables_in_C:
                    break_count = compute_break_count(x, clauses, assignment, variable_to_clauses)
                    break_counts[x] = break_count

                # Freebie move if any variable has break-count = 0
                zero_break_vars = [x for x in variables_in_C if break_counts[x] == 0]
                if zero_break_vars:
                    x = random.choice(zero_break_vars)
                    var_to_flip = x
                else:
                    if random.random() < p:
                        # Random walk move
                        var_to_flip = random.choice(variables_in_C)
                    else:
                        # Greedy move: variable with smallest break-count
                        min_break = min(break_counts.values())
                        min_break_vars = [x for x in variables_in_C if break_counts[x] == min_break]
                        var_to_flip = random.choice(min_break_vars)

                # Flip the chosen variable
                flip_variable(assignment, var_to_flip)

    return "FAIL"

def main():
    # Parse command-line arguments
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

    # Build the graph G and color it
    G = build_graph(clauses)
    colors = nx.coloring.greedy_color(G, strategy='largest_first')

    # Build variable to clauses mapping
    variable_to_clauses = {}
    for idx, clause in enumerate(clauses):
        for var in clause:
            x = abs(var)
            variable_to_clauses.setdefault(x, set()).add(idx)

    # Run Algorithm A1
    result = AlgorithmA1(clauses, colors, variable_to_clauses, max_tries, max_loops, probability)

    if result != "FAIL":
        print("Satisfying assignment found:")
        for var in sorted(result.keys()):
            val = result[var]
            print(f"Variable {var} = {val}")
    else:
        print("No satisfying assignment found within the given limits.")

if __name__ == "__main__":
    main()
