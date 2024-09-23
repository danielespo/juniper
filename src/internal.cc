#include "internal.hpp"
#include <fstream>
#define value(lit) (lit > 0 ? value[lit] : -value[-lit])    // Get the value of a literal
#define watch(id) (watches[vars + id])                      // Remapping a literal [-maxvar, +maxvar] to its watcher.


/*Goals
Make a SAT CDCL solver that is on par with CaDiCal for the Strukov Group in 2024



Features this has:

Removed the LBD decision heuristic, that made this much slower.

Need to copy the decision heuristic cadical uses! 

2 watched literals 
literal block distance heuristic
controls the restart frequency in restart();
BCP in propagate()

Features missing:

A lot of the extra heuristics in cadical

Most of them are useless, we are not doing incremental solving with this at all


Custom config to target UNSAT vs SAT ; Has to have frequent, aggresive restarts for UNSAT problems, long, focused search for SAT problems
Target phases rephasing heuristic using microwalksat.
Better output writing + DRAT proof! 


optimizations:
Has to interface nicely with SBVA, too
Need a cheap way to make an educated guess on its configuration when it starts running based on the local search preprocessing
has to use as much L0 cache as the system will let you

*/


char *read_whitespace(char *p) {                            // Aid function for parser
    while ((*p >= 9 && *p <= 13) || *p == 32) ++p;
    return p;
}

char *read_until_new_line(char *p) {                        // Aid function for parser
    while (*p != '\n') {
        if (*p++ == '\0') exit(1);
    }
    return ++p;
}

char *read_int(char *p, int *i) {                           // Aid function for parser
    bool sym = true; *i = 0;
    p = read_whitespace(p);
    if (*p == '-') sym = false, ++p;
    while (*p >= '0' && *p <= '9') {
        if (*p == '\0') return p;
        *i = *i * 10 + *p - '0', ++p;
    }
    if (!sym) *i = -(*i);
    return p;
}

int Solver::add_clause(std::vector<int> &c) {                   
    clause_DB.push_back(Clause(c.size()));                          // Add a clause c into database.
    int id = clause_DB.size() - 1;                                  // Getting clause index.
    for (int i = 0; i < (int)c.size(); i++) clause_DB[id][i] = c[i];     // Copy literals
    watch(-c[0]).push_back(Watcher(id, c[1]));                      // Watch this clause by literal -c[0]
    watch(-c[1]).push_back(Watcher(id, c[0]));                      // Watch this clause by literal -c[1]
    return id;                                                      
}

int Solver::parse(char *filename) {
    std::ifstream fin(filename);                                    // Fast load begin                                 
    fin.seekg(0, fin.end);
    size_t file_len = fin.tellg();
	fin.seekg(0, fin.beg);
	char *data = new char[file_len + 1], *p = data;
	fin.read(data, file_len);
	fin.close();                                                    // Fast load end
	data[file_len] = '\0';
    std::vector<int> buffer;                                        // Save the clause that waiting to push
    while (*p != '\0') {
        p = read_whitespace(p);
        if (*p == '\0') break;
        if (*p == 'c') p = read_until_new_line(p);
        else if (*p == 'p') {                                                               // Deal with 'p cnf' line.
            if (*(p + 1) == ' ' && *(p + 2) == 'c' && *(p + 3) == 'n' && *(p + 4) == 'f') {
                p += 5, p = read_int(p, &vars), p = read_int(p, &clauses);
                alloc_memory();
            } 
            else printf("PARSE ERROR! Unexpected char\n"), exit(2);                        // Wrong 'p ' line.
        }
        else {                                                                             
            int32_t dimacs_lit;
            p = read_int(p, &dimacs_lit);
            if (*p == '\0' && dimacs_lit != 0)                                              // Unexpected EOF
                printf("c PARSE ERROR! Unexpected EOF\n"), exit(1);
            if (dimacs_lit == 0) {                                                          // Finish read a clause.
                if (buffer.size() == 0) return 20;                                          // Read an empty clause.
                if (buffer.size() == 1 && value(buffer[0]) == -1) return 20;                // Found confliction in origin clauses
                if (buffer.size() == 1 && !value(buffer[0])) assign(buffer[0], 0, -1);      // Found an unit clause.
                else if (buffer.size() > 1) add_clause(buffer);                             // Found a clause who has more than 1 literals.
                buffer.clear();                                                             // For the next clause.
            }       
            else buffer.push_back(dimacs_lit);                                              // read a literal
        }
    }
    origin_clauses = clause_DB.size();
    return (propagate() == -1 ? 0 : 20);                                                    // Simplify by BCP.
}

void Solver::alloc_memory() { 
    // you used to be able to segfault here if you run it without an input....
    // I added (); to init empty

    value       = new int[vars + 1](); 
    reason      = new int[vars + 1]();
    level       = new int[vars + 1]();
    mark        = new int[vars + 1]();
    local_best  = new int[vars + 1]();
    saved       = new int[vars + 1]();
    activity    = new double[vars + 1]();
    watches     = new std::vector<Watcher>[vars * 2 + 1];
    conflicts = time_stamp = propagated = restarts = rephases = reduces = threshold = 0;
    
    // literal block distance heuristics removed
    
    // hardcoded rephase and reduce limits,  the function rephase() increases them
    var_inc = 1, rephase_limit = 1024, reduce_limit = 8192;
    vsids.setComp(GreaterActivity(activity));
    for (int i = 1; i <= vars; i++) 
        value[i] = reason[i] = level[i] = mark[i] = local_best[i] = activity[i] = saved[i] = 0, vsids.insert(i);
}

void Solver::bump_var(int var, double coeff) { // Increases the VSIDS activity score of a variable for the decision heuristic
    if ((activity[var] += var_inc * coeff) > 1e100) {           // Update score and prevent float overflow
        for (int i = 1; i <= vars; i++) activity[i] *= 1e-100;
        var_inc *= 1e-100;}
    if (vsids.inHeap(var)) vsids.update(var);                 // update heap
}

void Solver::assign(int lit, int l, int cref) {
    int var = abs(lit);
    value[var]  = lit > 0 ? 1 : -1; // Assigns a value to the literal’s corresponding variable.
    level[var]  = l, reason[var] = cref; // Records the decision level and reason for the assignment.                                         
    trail.push_back(lit); // Adds literal to the trail (a record of all assignments made).
}

int Solver::propagate() { // Propagates the implications of a new assignment.
    while (propagated < (int)trail.size()) { 

        int p = trail[propagated++];                    // Pick an unpropagated literal in the trail.
        
        std::vector<Watcher> &ws = watch(p);            // Fetch the watcher for this literal (2 watched literals)
        int i, j, size = ws.size();                     
        for (i = j = 0; i < size; ) {               
            int blocker = ws[i].blocker;                       
            if (value(blocker) == 1) {                  // Pre-judge whether the clause is already SAT
                ws[j++] = ws[i++]; continue;
            }
            int cref = ws[i].idx_clause, k, sz;
            Clause& c = clause_DB[cref];                // Fetch a clause from watcher
            if (c[0] == -p) c[0] = c[1], c[1] = -p;     // Make sure c[1] is the false literal (-p)
            Watcher w = Watcher(cref, c[0]);            // Prepare a new watcher for c[1]
            i++;
            if (value(c[0]) == 1) {                     // Check whether another lit is SAT.
                ws[j++] = w; continue;
            }
            for (k = 2, sz = c.lit.size(); k < sz && value(c[k]) == -1; k++);    // Find a new watch literal
            if (k < sz) {                               // Move the watch literal to the second place
                c[1] = c[k], c[k] = -p;
                watch(-c[1]).push_back(w);
            }
            else {                                      // Can not find a new watch literal
                ws[j++] = w;
                if (value(c[0]) == -1) {                // There is a conflict
                    while (i < size) ws[j++] = ws[i++];
                    ws.resize(j);
                    return cref;
                }
                else assign(c[0], level[abs(p)], cref); // Find a new unit clause and assign it
            }
        }
        ws.resize(j);
    }
    return -1;                                          // Meet a convergence = (unique implication point UIP)
}

int Solver::analyze(int conflict, int &backtrackLevel) {
    ++time_stamp;
    learnt.clear();
    Clause &c = clause_DB[conflict]; 
    int highestLevel = level[abs(c[0])];
    if (highestLevel == 0) return 20;
    learnt.push_back(0);        // leave a place to save the First-UIP
    std::vector<int> bump;      // The variables to bump
    int should_visit_ct = 0,    // The number of literals that have not been visited in the higest level of the implication graph.
        resolve_lit = 0,        // The literal to do resolution.
        index = trail.size() - 1;
    do {
        Clause &c = clause_DB[conflict];
        for (int i = (resolve_lit == 0 ? 0 : 1); i < (int)c.lit.size(); i++) {
            int var = abs(c[i]);
            if (mark[var] != time_stamp && level[var] > 0) {
                bump_var(var, 0.5);
                bump.push_back(var);
                mark[var] = time_stamp;
                if (level[var] >= highestLevel) should_visit_ct++;
                else learnt.push_back(c[i]);
            }
        }
        do {                                         // Find the last marked literal in the trail to do resolution.
            while (mark[abs(trail[index--])] != time_stamp);
            resolve_lit = trail[index + 1];
        } while (level[abs(resolve_lit)] < highestLevel);
        conflict = reason[abs(resolve_lit)], mark[abs(resolve_lit)] = 0, should_visit_ct--;
    } while (should_visit_ct > 0);                   // Have find the convergence node in the highest level (First UIP)
    learnt[0] = -resolve_lit;


    if (learnt.size() == 1) backtrackLevel = 0;
    else {                                           // find the second highest level for backtracking.
        int max_id = 1;
        for (int i = 2; i < (int)learnt.size(); i++)
            if (level[abs(learnt[i])] > level[abs(learnt[max_id])]) max_id = i;
        int p = learnt[max_id];
        learnt[max_id] = learnt[1], learnt[1] = p, backtrackLevel = level[abs(p)];
    }
    for (int i = 0; i < (int)bump.size(); i++)       // heuristically bump some variables.
        if (level[bump[i]] >= backtrackLevel - 1) bump_var(bump[i], 1);
    return 0;
}

void Solver::backtrack(int backtrackLevel) {
    if ((int)pos_in_trail.size() <= backtrackLevel) return;
    for (int i = trail.size() - 1; i >= pos_in_trail[backtrackLevel]; i--) {
        int v = abs(trail[i]);
        value[v] = 0, saved[v] = trail[i] > 0 ? 1 : -1; // phase saving 
        if (!vsids.inHeap(v)) vsids.insert(v);          // update heap
    }
    propagated = pos_in_trail[backtrackLevel];
    trail.resize(propagated);
    pos_in_trail.resize(backtrackLevel);
}

int Solver::decide() {      
    int next = -1;
    while (next == -1 || value(next) != 0) {    // Picking a variable according to VSIDS
        if (vsids.empty()) return 10;
        else next = vsids.pop();
    }
    pos_in_trail.push_back(trail.size());
    if (saved[next]) next *= saved[next];       // Pick the polarity of the varible
    assign(next, pos_in_trail.size(), -1);
    return 0;
}

void Solver::restart() {
    backtrack(0);
    int phase_rand = rand() % 100;              // probabilistic rephasing
    if ((phase_rand -= 60) < 0)     for (int i = 1; i <= vars; i++) saved[i] = local_best[i];
    else if ((phase_rand -= 5) < 0) for (int i = 1; i <= vars; i++) saved[i] = -local_best[i];
    else if ((phase_rand -= 20) < 0)for (int i = 1; i <= vars; i++) saved[i] = rand() % 2 ? 1 : -1;
}

void Solver::rephase() {
    rephases = 0, threshold *= 0.9, rephase_limit += 8192;
}

void Solver::reduce() {
    backtrack(0);
    reduces = 0, reduce_limit += 512;
    int new_size = origin_clauses, old_size = clause_DB.size();
    reduce_map.resize(old_size);

    clause_DB.resize(new_size, Clause(0));
    for (int v = -vars; v <= vars; v++) {   // Update the watches.
        if (v == 0) continue;
        int old_sz = watch(v).size(), new_sz = 0;
        for (int i = 0; i < old_sz; i++) {
            int old_idx = watch(v)[i].idx_clause;
            int new_idx = old_idx < origin_clauses ? old_idx : reduce_map[old_idx];
            if (new_idx != -1) {
                watch(v)[i].idx_clause = new_idx;
                if (new_sz != i) watch(v)[new_sz] = watch(v)[i];
                new_sz++;
            }
        }
        watch(v).resize(new_sz);
    }
}

int Solver::solve() {
    int res = 0;
    while (!res) {
        int cref = propagate();                         // Boolean Constraint Propagation (BCP)
        if (cref != -1) {                               // Find a conflict
            int backtrackLevel = 0;
            res = analyze(cref, backtrackLevel);   // Conflict analyze
            if (res == 20) break;                       // Find a conflict in 0-level
            backtrack(backtrackLevel);                  // backtracking         
            if (learnt.size() == 1) assign(learnt[0], 0, -1);   // Learnt a unit clause.
            else {                     
                int cref = add_clause(learnt);                  // Add a clause to data base.           
                assign(learnt[0], backtrackLevel, cref);        // The learnt clause implies the assignment of the UIP variable.
            }
            var_inc *= (1 / 0.8);                               // var_decay for locality
            ++restarts, ++conflicts, ++rephases, ++reduces;     
            if ((int)trail.size() > threshold) {                // update the local-best phase
                threshold = trail.size();                       
                for (int i = 1; i <= vars; i++) local_best[i] = value[i];
            }
        }
        else if (reduces >= reduce_limit) reduce();     
        // note, no restart condition here anymore since it used to be LBD controlled       
        else if (rephases >= rephase_limit) rephase();
        else res = decide();
    }
    return res;
}

void Solver::printModel() {
    printf("v ");
    for (int i = 1; i <= vars; i++) printf("%d ", value[i] * i);
    puts("0");
}

int main(int argc, char **argv) {
    Solver S;
    int res = S.parse(argv[1]);
    if (res == 20) printf("s UNSATISFIABLE\n");
    else {
        res = S.solve();
        if (res == 10) {
            printf("s SATISFIABLE\n");
            S.printModel();
        }
        else if (res == 20) printf("s UNSATISFIABLE\n");
    }
    return 0;
}