import random
import argparse
import time

"""

1) Choose cc UNSAT clauses (instead of one).
2) Follow the same algorithm as in WalkSAT to determine cc candidate variables to flip, say set cc_candidates_to_flip (unsat variables to flip)
3) From  cc_candidates_to_flip, choose a subset of uncorrelated (i.e., from the same color) variables to flip. 
The color can be chosen randomly from those in  cc_candidates_to_flip, or the color with the largest number of variables 
in cc_candidates_to_flip and/or based on the rotation of colors

# NOTE: A1 is going to be more optimal than A2 most likely
"""

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

# Regular WalkSAT with no frills
def walkSAT(clauses, max_tries, max_flips, p):
    def evaluate_clause(clause, assignment):
        return any((var > 0 and assignment.get(abs(var), False)) or 
                   (var < 0 and not assignment.get(abs(var), False)) for var in clause)

    def get_unsatisfied_clauses(clauses, assignment):
        return [clause for clause in clauses if not evaluate_clause(clause, assignment)]

    def get_variables(clauses):
        return set(abs(var) for clause in clauses for var in clause)

    def flip_variable(assignment, var):
        assignment[var] = not assignment[var]

    for _ in range(max_tries):
        variables = list(get_variables(clauses))
        assignment = {var: random.choice([True, False]) for var in variables}
        
        for _ in range(max_flips):

            unsatisfied = get_unsatisfied_clauses(clauses, assignment)
            if not unsatisfied:
                return assignment  # Found a satisfying assignment
            
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

    return "FAIL"

def main():
    parser = argparse.ArgumentParser(description='WalkSAT Solver (regular).')
    parser.add_argument('-cnf', type=str,help='Path to SAT problem in .cnf format', required=True)
    parser.add_argument('-p', type=float, help='Probability float between 0 and 1', required=True)
    parser.add_argument('--max_tries', type=int, default=100, help='Maximum number of tries')
    parser.add_argument('--max_flips', type=int, default=1000, help='Maximum number of loops per try')
    args = parser.parse_args()

    filepath = args.cnf
    probability = args.p
    max_tries = args.max_tries
    max_flips = args.max_flips

    # Read and preprocess the CNF file
    try:
        num_vars, clauses = read_dimacs(filepath)
    except Exception as e:
        print(f"Error reading CNF file: {e}")
        return

    # Running WalkSAT
    start_walksat_time = time.perf_counter()
    result = walkSAT(clauses, max_tries, max_flips, probability)
    end_walksat_time = time.perf_counter()

    time_walksat = end_walksat_time - start_walksat_time

    if result != "FAIL":
        print("SAT:")
        for var in sorted(result):
            print(f"Variable {var} : {result[var]}")
    else:
        print("No satisfying assignment found within the given limits.")

    print(f"WalkSAT completed in {time_walksat:.4f} seconds.")

if __name__ == "__main__":
    main()