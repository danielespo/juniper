import random
import time
from collections import defaultdict

# Little pet project: can GPT o1-preview replicate an algorithm
# with no human intervention and accrue something practically useful?

# Answer: No.
# Real answer: this is not half bad, it came up with greedy coloring on its
# own, likely plagiarized from the networkX package. Not very interesting,
# and does not compare in the benchmark to what we are using :)


def read_dimacs(filename):
    clauses = []
    num_vars = 0
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

def walksat_color(F, max_tries=1000, max_loops=1000, p=0.5):
    stats = {
        'total_time': 0,
        'greedy_coloring_time': 0,
        'tries': 0,
        'loops': 0,
        'freebie_moves': 0,
        'random_walk_moves': 0,
        'greedy_moves': 0,
        'flips': 0,
        'break_counts_computed': 0,
        'colors_used': 0,
        'clauses_evaluated': 0,
        'unsatisfied_clauses': [],
        'satisfying_assignment_found': False
    }
    start_time = time.time()

    # Build G: Graph of literals with edges between literals that appear together in a clause
    build_start_time = time.time()
    G = defaultdict(set)  # adjacency list
    K = set()  # set of literals
    clauses_with_literal = defaultdict(set)  # clauses containing each literal
    clauses_with_var = defaultdict(set)  # clauses containing each variable (positive or negative)

    for idx, clause in enumerate(F):
        literals = clause
        for l in literals:
            K.add(l)
            clauses_with_literal[l].add(idx)
            clauses_with_var[abs(l)].add(idx)
        for l1 in literals:
            for l2 in literals:
                if l1 != l2:
                    G[l1].add(l2)
    build_end_time = time.time()
    stats['graph_building_time'] = build_end_time - build_start_time

    # Greedy Coloring
    coloring_start_time = time.time()
    g = greedy_color(G, K)
    coloring_end_time = time.time()
    stats['greedy_coloring_time'] = coloring_end_time - coloring_start_time
    c = set(g.values())
    stats['colors_used'] = len(c)

    # Precompute variables of each color
    color_vars = defaultdict(set)
    for literal, color in g.items():
        color_vars[color].add(literal)

    # For each try
    for i in range(max_tries):
        stats['tries'] += 1
        # Randomly initialize assignment σ
        variables = set(abs(l) for l in K)
        σ = {x: random.choice([True, False]) for x in variables}

        # For each loop
        for j in range(max_loops):
            stats['loops'] += 1
            # Check if σ satisfies F
            if is_satisfied(F, σ, stats):
                stats['total_time'] = time.time() - start_time
                stats['satisfying_assignment_found'] = True
                return σ, stats  # success

            # For each color v in c
            for color in c:
                # f ← clauses in F with at least one variable of the current color v
                literals_of_color = color_vars[color]
                clauses_in_f = set()
                for l in literals_of_color:
                    clauses_in_f.update(clauses_with_literal[l])

                # Get unsatisfied clauses in f
                unsatisfied_clauses = []
                for clause_idx in clauses_in_f:
                    clause = F[clause_idx]
                    stats['clauses_evaluated'] += 1
                    if not is_clause_satisfied(clause, σ):
                        unsatisfied_clauses.append((clause_idx, clause))

                if not unsatisfied_clauses:
                    continue  # No unsatisfied clauses for this color

                # Choose C randomly from unsatisfied clauses in f
                C_idx, C = random.choice(unsatisfied_clauses)
                stats['unsatisfied_clauses'].append(C_idx)

                # Variables x ∈ C with color = current color
                variables_in_C = [l for l in C if g[l] == color]
                # Compute break-counts for these variables
                break_counts = {}
                for l in variables_in_C:
                    x = abs(l)
                    bc = compute_break_count(l, σ, F, clauses_with_var)
                    break_counts[l] = bc
                    stats['break_counts_computed'] += 1

                # Check if any x has break-count = 0
                zero_break_vars = [l for l in break_counts if break_counts[l] == 0]
                if zero_break_vars:
                    # Freebie move: pick any x with break-count 0
                    v = random.choice(zero_break_vars)
                    stats['freebie_moves'] += 1
                else:
                    # With probability p, pick random variable in C with color = c
                    if random.random() < p:
                        v = random.choice(variables_in_C)
                        stats['random_walk_moves'] += 1
                    else:
                        # Greedy move: pick variable with smallest break-count
                        min_break = min(break_counts.values())
                        min_break_vars = [l for l in break_counts if break_counts[l] == min_break]
                        v = random.choice(min_break_vars)
                        stats['greedy_moves'] += 1

                # Flip v in σ
                x = abs(v)
                σ[x] = not σ[x]
                stats['flips'] += 1
        # End of loops
    # End of tries
    stats['total_time'] = time.time() - start_time
    return None, stats  # FAIL

def greedy_color(G, K):
    g = {}  # mapping from literal to color
    for v in K:
        used_colors = set()
        for neighbor in G[v]:
            if neighbor in g:
                used_colors.add(g[neighbor])
        # Assign smallest possible color
        color = 0
        while color in used_colors:
            color += 1
        g[v] = color
    return g

def is_satisfied(F, σ, stats=None):
    for clause in F:
        if not is_clause_satisfied(clause, σ):
            return False
    return True

def is_clause_satisfied(clause, σ):
    for l in clause:
        x = abs(l)
        val = σ[x]
        if l < 0:
            val = not val
        if val:
            return True
    return False

def compute_break_count(literal, σ, F, clauses_with_var):
    x = abs(literal)
    break_count = 0
    original_value = σ[x]
    σ[x] = not σ[x]  # Flip x temporarily
    for clause_idx in clauses_with_var[x]:
        clause = F[clause_idx]
        before = is_clause_satisfied(clause, σ)
        σ[x] = not σ[x]  # Flip back to original value
        after = is_clause_satisfied(clause, σ)
        σ[x] = not σ[x]  # Flip again to temporary value
        if before and not after:
            break_count += 1
        elif not before and after:
            break_count -= 1
    σ[x] = original_value  # Restore x
    return break_count


if __name__ == "__main__":

    num_vars, F = read_dimacs('easiestcnf.cnf')
    result, stats = walksat_color(F, max_tries=1000, max_loops=1000, p=0.5)

    if result:
        print("Satisfying assignment found:")
        for var in sorted(result):
            print(f"x{var} = {result[var]}")
    else:
        print("No satisfying assignment found.")

    # Print the collected statistics
    print("\nAlgorithm Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
