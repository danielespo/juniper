import networkx as nx
import argparse

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
                # Remove trailing 0 if there
                if clause and clause[-1] == 0:
                    clause = clause[:-1]
                if clause:
                    clauses.append(clause)
    return num_vars, clauses

def GenerateColors(clauses):
    variables = set(abs(literal) for clause in clauses for literal in clause)
    G = nx.Graph()
    G.add_nodes_from(variables)  
    for clause in clauses:
        vars_in_clause = set(abs(literal) for literal in clause)
        for var1 in vars_in_clause:
            for var2 in vars_in_clause:
                if var1 != var2:
                    G.add_edge(var1, var2)

    # Greedy coloring, return dictionary of unique colors (0 to n) for each node
    colors = nx.coloring.greedy_color(G, strategy='largest_first')
    return colors

def main():
    parser = argparse.ArgumentParser(description='Generate colors for variables using NetworkX.')
    parser.add_argument('-cnf', help='Path to SAT problem in .cnf format', required=True)
    parser.add_argument('-out', help='Output file for colors', required=True)
    args = parser.parse_args()

    filepath = args.cnf
    output_file = args.out

    try:
        num_vars, clauses = read_dimacs(filepath)
    except Exception as e:
        print(f"Error reading CNF file: {e}")
        raise ValueError

    colors = GenerateColors(clauses)

    # Save colors to a file
    with open(output_file, 'w') as f:
        for var, color in colors.items():
            f.write(f"{var} {color}\n")

if __name__ == "__main__":
    main()
