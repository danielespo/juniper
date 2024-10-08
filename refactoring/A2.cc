#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include <map>
#include <random>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <tuple>
#include <future>
#include <limits>

// Daniel Espinosa, Strukov Group, UC Santa Barbara 2024
// A2 Algorithm
// Refactored to C++
// Algorithmically equivalent to wsatA2.py
// Much faster in wall clock time, this one is also multithreaded.

// Compile:
// g++ -std=c++11 -O2 -o A2 A2.cc -pthread
// Use: 
// python colorandtest.py -cnf path_to_cnf_file.cnf -out colors.txt
// ./A2 -cnf path_to_cnf_file.cnf -colors colors.txt -p 0.5 --max_tries 100 --max_loops 1000

bool read_dimacs(const std::string& filename, int& num_vars, std::vector<std::vector<int>>& clauses) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error opening CNF file: " << filename << std::endl;
        return false;
    }
    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        if (line[0] == 'p') {
            std::istringstream iss(line);
            std::string tmp;
            iss >> tmp; // skip 'p'
            iss >> tmp; // skip 'cnf'
            iss >> num_vars;
            int num_clauses;
            iss >> num_clauses;
        } else if (line[0] == 'c' || line[0] == '%' || line[0] == '0') {
            continue;
        } else {
            std::istringstream iss(line);
            std::vector<int> clause;
            int lit;
            while (iss >> lit) {
                if (lit == 0) break;
                clause.push_back(lit);
            }
            if (!clause.empty()) {
                clauses.push_back(clause);
            }
        }
    }
    return true;
}

// Read colors from file generated by Python script
bool read_colors(const std::string& filename, std::unordered_map<int, int>& colors) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error opening colors file: " << filename << std::endl;
        return false;
    }
    int var, color;
    while (infile >> var >> color) {
        colors[var] = color;
    }
    return true;
}

// Evaluate a clause given the current assignment
bool evaluate_clause(const std::vector<int>& clause, const std::unordered_map<int, bool>& assignment) {
    for (int var : clause) {
        int var_abs = std::abs(var);
        bool var_value = assignment.at(var_abs);
        if ((var > 0 && var_value) || (var < 0 && !var_value)) {
            return true;
        }
    }
    return false;
}

// Get all unsatisfied clauses
std::vector<std::vector<int>> get_unsatisfied_clauses(
    const std::vector<std::vector<int>>& clauses,
    const std::unordered_map<int, bool>& assignment) {
    std::vector<std::vector<int>> unsatisfied;
    for (const auto& clause : clauses) {
        if (!evaluate_clause(clause, assignment)) {
            unsatisfied.push_back(clause);
        }
    }
    return unsatisfied;
}

// Get all variables from the clauses
std::unordered_set<int> get_variables(const std::vector<std::vector<int>>& clauses) {
    std::unordered_set<int> variables;
    for (const auto& clause : clauses) {
        for (int var : clause) {
            variables.insert(std::abs(var));
        }
    }
    return variables;
}

// Flip the value of a variable in the assignment
void flip_variable(std::unordered_map<int, bool>& assignment, int var) {
    assignment[var] = !assignment[var];
}

// Main algorithm (A2)
// Steps:
// 1) Iterate over different colors
// 2) Choose cc clauses from C (i.e., cc UNSAT clauses, each with one variable of v color)
// 3) Follow the same algorithm as in WalkSAT to determine cc candidate variables to flip, say set cc_candidates_to_flip
// 4) In cc_candidates_to_flip, pick only variables of v color to flip

std::tuple<std::unordered_map<int, bool>, int, int, int> AlgorithmA2(
    const std::vector<std::vector<int>>& clauses,
    const std::unordered_map<int, int>& colors,
    int max_tries,
    int max_loops,
    double p) {

    int flips = 0;
    std::vector<int> variables_vec;
    for (const auto& var : get_variables(clauses)) {
        variables_vec.push_back(var);
    }
    // Seedless rand
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::uniform_int_distribution<> bool_dis(0, 1);

    // Get the set of colors
    std::unordered_set<int> color_set; //always return a non-const type
    for (const auto& kv : colors) {
        color_set.insert(kv.second);
    } 

    // Build variable to clauses mapping
    std::unordered_map<int, std::vector<int>> variable_to_clauses;
    for (size_t clause_idx = 0; clause_idx < clauses.size(); ++clause_idx) {
        const auto& clause = clauses[clause_idx];
        for (int var : clause) {
            variable_to_clauses[std::abs(var)].push_back(clause_idx);
        }
    }

    for (int _try = 0; _try < max_tries; ++_try) {
        // Initialize random assignment
        std::unordered_map<int, bool> assignment;
        for (int var : variables_vec) {
            assignment[var] = bool_dis(gen);
        }

        // Initialize clause satisfaction status
        std::vector<bool> clause_satisfied(clauses.size());
        for (size_t i = 0; i < clauses.size(); ++i) {
            clause_satisfied[i] = evaluate_clause(clauses[i], assignment);
        }

        for (int _loop = 0; _loop < max_loops; ++_loop) {
            // Get unsatisfied clauses
            std::vector<int> unsat_clause_indices;
            for (size_t i = 0; i < clauses.size(); ++i) {
                if (!clause_satisfied[i]) {
                    unsat_clause_indices.push_back(i);
                }
            }

            if (unsat_clause_indices.empty()) {
                return std::make_tuple(assignment, _try, _loop, flips); // Success
            }

            // Step 1: Iterate over different colors
            // A1: const auto& cc = unsat_clause_indices;
            
            bool variable_flipped = false;
            
            // Iterate over each color
            for (int current_color : color_set) {
                // Step 2: Choose unsatisfied clauses that contain at least one variable of current_color
                std::vector<int> cc; // Indices of clauses
                for (int clause_idx : unsat_clause_indices) {
                    const auto& clause = clauses[clause_idx];
                    for (int var : clause) {
                        if (colors.at(std::abs(var)) == current_color) {
                            cc.push_back(clause_idx);
                            break;
                        }
                    }
                }

                if (cc.empty()) {
                    continue; // No unsatisfied clauses with this color
                }

                // Collect all unique variables of current_color from these clauses
                std::unordered_set<int> variables_to_process;
                for (int clause_idx : cc) {
                    const auto& clause = clauses[clause_idx];
                    for (int var : clause) {
                        int var_abs = std::abs(var);
                        if (colors.at(var_abs) == current_color) {
                            variables_to_process.insert(var_abs);
                        }
                    }
                }

                if (variables_to_process.empty()) {
                    continue; // No variables of current_color to process
                }

                // Compute break counts for variables_to_process in parallel
                // https://tamerlan.dev/introduction-to-pthreads/
                std::vector<std::future<std::pair<int, int>>> futures;
                for (int x : variables_to_process) {
                    // Compute the break counts in parallel lazily when we need them
                    futures.push_back(std::async(std::launch::async, [&, x]() {
                        int break_count = 0;

                        // Flip x temporarily
                        flip_variable(assignment, x);

                        // Compute break count
                        for (int c_idx : variable_to_clauses[x]) {
                            if (clause_satisfied[c_idx]) {
                                if (!evaluate_clause(clauses[c_idx], assignment)) {
                                    break_count++;
                                }
                            }
                        }

                        // Flip x back
                        flip_variable(assignment, x);

                        return std::make_pair(x, break_count);
                    }));
                }

                std::vector<std::tuple<int, int, int>> cc_candidates_to_flip;

                for (auto& future : futures) {
                    try {
                        auto result = future.get();
                        int x = result.first;
                        int break_count = result.second;
                        cc_candidates_to_flip.emplace_back(x, break_count, current_color);
                    } catch (const std::exception& e) {
                        std::cerr << "Error computing break count: " << e.what() << std::endl;
                        continue;
                    }
                }

                if (cc_candidates_to_flip.empty()) {
                    continue; // No candidates to flip
                }

                // Select variables to flip from the candidates with minimum break-count
                int min_break_count = std::numeric_limits<int>::max();
                for (const auto& item : cc_candidates_to_flip) {
                    int break_count = std::get<1>(item);
                    if (break_count < min_break_count) {
                        min_break_count = break_count;
                    }
                }

                std::vector<int> vars_with_min_break;
                for (const auto& item : cc_candidates_to_flip) {
                    if (std::get<1>(item) == min_break_count) {
                        vars_with_min_break.push_back(std::get<0>(item));
                    }
                }

                if (vars_with_min_break.empty()) {
                    continue; // No variables to flip
                }

                int var_to_flip;
                if (dis(gen) < p) {
                    // Random walk move: flip a random variable from the candidates
                    std::uniform_int_distribution<> idx_dis(0, vars_with_min_break.size() - 1);
                    int idx = idx_dis(gen);
                    var_to_flip = vars_with_min_break[idx];
                } else {
                    // Greedy move: flip the variable with the smallest break-count
                    std::uniform_int_distribution<> idx_dis(0, vars_with_min_break.size() - 1);
                    int idx = idx_dis(gen);
                    var_to_flip = vars_with_min_break[idx];
                }

                // Flip the variable and update clause satisfaction status
                flip_variable(assignment, var_to_flip);
                flips++;

                // Update clause satisfaction status
                for (int c_idx : variable_to_clauses[var_to_flip]) {
                    clause_satisfied[c_idx] = evaluate_clause(clauses[c_idx], assignment);
                }

                variable_flipped = true;

                // Break after flipping a variable for the current color to go to the next loop iteration
                break;
            }

            if (!variable_flipped) {
                // If no color led to a flip, continue to next loop iteration
                continue;
            }
        }
    }

    // Return failure
    return std::make_tuple(std::unordered_map<int, bool>(), -1, -1, flips);
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string filepath;
    std::string colors_file;
    double probability = 0.5;
    int max_tries = 100;
    int max_loops = 1000;

    // Simple command line parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-cnf" && i + 1 < argc) {
            filepath = argv[++i];
        } else if (arg == "-colors" && i + 1 < argc) {
            colors_file = argv[++i];
        } else if (arg == "-p" && i + 1 < argc) {
            probability = std::stod(argv[++i]);
        } else if (arg == "--max_tries" && i + 1 < argc) {
            max_tries = std::stoi(argv[++i]);
        } else if (arg == "--max_loops" && i + 1 < argc) {
            max_loops = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return 1;
        }
    }

    if (filepath.empty() || colors_file.empty()) {
        std::cerr << "Usage: " << argv[0] << " -cnf filename -colors colors_file -p probability [--max_tries N] [--max_loops N]" << std::endl;
        return 1;
    }

    int num_vars;
    std::vector<std::vector<int>> clauses;

    if (!read_dimacs(filepath, num_vars, clauses)) {
        std::cerr << "Error reading CNF file." << std::endl;
        return 1;
    }

    std::unordered_map<int, int> colors;
    if (!read_colors(colors_file, colors)) {
        std::cerr << "Error reading colors file." << std::endl;
        return 1;
    }

    auto start_algo_time = std::chrono::high_resolution_clock::now();
    auto result = AlgorithmA2(clauses, colors, max_tries, max_loops, probability);
    auto end_algo_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_algo = end_algo_time - start_algo_time;

    if (std::get<1>(result) != -1) {
        // Success
        int tries = std::get<1>(result);
        int loops = std::get<2>(result);
        int flips = std::get<3>(result);
        std::cout << time_algo.count() << " " << 0 << " " << tries << " " << loops << " " << flips << std::endl;
    } else {
        // Fail
        std::cout << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 << std::endl;
    }

    return 0;
}
