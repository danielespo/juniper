import random
import argparse
import networkx as nx
import time
import sys
from itertools import combinations

# 1.1: networkX set(G) likes creating a fake node 0 which we have to get rid of
# I was not getting rid of it and hence it kept picking 0 and well, stalling

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

# A2 Algorithm
def ColorWalkSAT(clauses, colors, max_tries, max_loops, p):

    # Same as before
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
    
    color_vars = {}
    variables = list(get_variables(clauses))
    for var in variables:
        color = colors.get(var, None)
        if color is not None:
            color_vars.setdefault(color, []).append(var)
        else:
            raise ValueError("All  variables should have a color, did NetworkX fail?")
            #color_vars.setdefault('no_color', []).append(var) # Colorless variables, future version (have regular wsat in parallel)

    for _ in range(max_tries):
        variables = list(get_variables(clauses))
        assignment = {var: random.choice([True, False]) for var in variables}

        for _ in range(max_loops):
            unsatisfied = get_unsatisfied_clauses(clauses, assignment)
            
            if not unsatisfied:
                return assignment  # Found a satisfying assignment

            # Iterate over colors
            for color in color_vars:
                
                # Doing this again as a safety
                unsatisfied = get_unsatisfied_clauses(clauses, assignment) # can check len unsat to see what's going on
                vars_in_color = color_vars[color]
                # Find unsatisfied clauses involving variables of this color
                unsat_clauses_in_color = [
                    clause for clause in unsatisfied
                    if any(abs(var) in vars_in_color for var in clause)
                ]

                if not unsat_clauses_in_color:
                    break  # No unsatisfied clauses involving this color

                # Randomly select one unsatisfied clause which has this color present (at least one literal has the color)
                clause = random.choice(unsat_clauses_in_color)

                if random.random() < p:
                    # Flip a the variable from the clause iff it also has the given color.
                    # each clause has only one variable of each color
                    vars_in_clause_and_color = [
                        abs(var) for var in clause if abs(var) in vars_in_color
                    ]
                    var_to_flip = vars_in_clause_and_color
                else:
                    # Flip a variable that minimizes the number of unsatisfied clauses if flipped
                    # now whiteboard version
                    """
                    Traceback (most recent call last):
                    File "/home/dae/SatExperiments/juniper/wsatA2.py", line 176, in <module>
                        main()
                    File "/home/dae/SatExperiments/juniper/wsatA2.py", line 164, in main
                        result = ColorWalkSAT(clauses, colors, max_tries, max_loops, probability)
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    File "/home/dae/SatExperiments/juniper/wsatA2.py", line 113, in ColorWalkSAT
                        flip_variable(assignment, var_to_flip) #can either be random or not
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    File "/home/dae/SatExperiments/juniper/wsatA2.py", line 51, in flip_variable
                        assignment[var] = not assignment[var]
                                            ~~~~~~~~~~^^^^^
                    TypeError: unhashable type: 'list'""" # TODO fix this obnoxious bug

                    break_counts = []
                    for clause in unsat_clauses_in_color:
                        for var in clause:
                            if abs(var) in vars_in_color:
                                assignment[abs(var)] = not assignment[abs(var)]
                                break_counts.append((len(get_unsatisfied_clauses(clauses, assignment)), abs(var)))
                                assignment[abs(var)] = not assignment[abs(var)]  # Undo the flip
                    
                    min_break = min(break_counts, key=lambda x: x[0])
                    vars_with_min_break = [var for break_count, var in break_counts if break_count == min_break[0]]
                    var_to_flip = random.choice(vars_with_min_break)
                
                flip_variable(assignment, var_to_flip) #can either be random or not

    return "FAIL"

def main():
    
    parser = argparse.ArgumentParser(description='ColoringWalksat SAT Solver.')
    parser.add_argument('-cnf', help='Path to SAT problem in .cnf format', required=True)
    parser.add_argument('-p', type=float, help='Probability float between 0 and 1', required=True)
    parser.add_argument('--max_tries', type=int, default=100, help='Maximum number of tries')
    parser.add_argument('--max_loops', type=int, default=1000, help='Maximum number of loops per try')
    args = parser.parse_args()

    filepath = args.cnf
    probability = args.p
    max_tries = args.max_tries
    max_loops = args.max_loops

    # Read and preprocess the CNF file
    try:
        num_vars, clauses = read_dimacs(filepath)
    except Exception as e:
        print(f"Error reading CNF file: {e}")
        raise ValueError

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

    # Greedy coloring , returns dictionary of unique colors (0 to n) for each node
    colors = nx.coloring.greedy_color(G, strategy='largest_first')
    end_color_time = time.perf_counter()
    
    number_of_colors = len(set(colors.values()))
    print("GraphColoring found "+ str(number_of_colors) +" colors")
    time_color = end_color_time - start_color_time

    # Running WalkSAT

    start_colorwalksat_process_time = time.perf_counter()
    result = ColorWalkSAT(clauses, colors, max_tries, max_loops, probability)
    end_colorwalksat_process_time = time.perf_counter()

    time_colorwalksat = end_colorwalksat_process_time - start_colorwalksat_process_time

    if result != "FAIL":
        SAT = 1
        print(time_colorwalksat, time_color, SAT)
    else:
        print("No satisfying assignment found within the given limits.")

if __name__ == "__main__":
    main()