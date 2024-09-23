#include "heap.hpp"

class Clause {
public:
    std::vector<int> lit;  
    Clause(int sz) { lit.resize(sz);}
    int& operator[] (int index) { return lit[index];}
};

class Watcher {
public:
    int idx_clause              // The clause index in clause database.
        , blocker;              // Used to fast guess whether a clause is already satisfied. 
    Watcher(): idx_clause(0), blocker(0) {}
    Watcher(int c, int b): idx_clause(c), blocker(b) {}
};

struct GreaterActivity {        // A compare function used to sort the activities.
    const double *activity;     
    bool operator() (int a, int b) const { return activity[a] > activity[b]; }
    GreaterActivity(): activity(NULL) {}
    GreaterActivity(const double *s): activity(s) {}
};

class Solver {
public:
    std::vector<int>    learnt,                     // The clause indices of the learnt clauses.
                        trail,                      // Save the assigned literal sequence.
                        pos_in_trail,               // Save the decision variables' position in trail.
                        reduce_map;                 // Auxiliary data structure for clause management.
    std::vector<Clause> clause_DB;                  // clause database.
    std::vector<Watcher> *watches;                  // A mapping from literal to clauses.
    int vars, clauses, origin_clauses, conflicts;   // the number of variables, clauses, conflicts.
    int restarts, rephases, reduces;                // the number of conflicts since the last ... .
    int rephase_limit, reduce_limit;                // parameters for when to conduct rephase and reduce.
    int threshold;                                  // A threshold for updating the local_best phase.
    int propagated;                                 // The number of propagted literals in trail.
    int time_stamp;                                 // Aid parameter for conflict analyzation and LBD calculation.   
   
    int *value,                                     // The variable assignement (1:True; -1:False; 0:Undefine) 
        *reason,                                    // The index of the clause that implies the variable assignment.
        *level,                                     // The decision level of a variable      
        *mark,                                      // Aid for conflict analyzation.
        *local_best,                                // A phase with a local deepest trail.                     
        *saved;                                     // Phase saving.
    double *activity;                               // The variables' score for VSIDS.   
    double var_inc;                                 // Parameter for VSIDS.               
    Heap<GreaterActivity> vsids;                    // Heap to select variable.
     
    void alloc_memory();                                    // Allocate memory for EasySAT 
    void assign(int lit, int level, int cref);              // Assigned a variable.
    int  propagate();                                       // BCP
    void backtrack(int backtrack_level);                    // Backtracking
    int  analyze(int cref, int &backtrack_level); // Conflict analyzation.
    int  parse(char *filename);                             // Read CNF file.
    int  solve();                                           // Solving.
    int  decide();                                          // Pick desicion variable.
    int  add_clause(std::vector<int> &c);                    // add new clause to clause database.
    void bump_var(int var, double mult);                     // update activity      
    void restart();                                         // do restart.                                      
    void reduce();                                          // do clause management.
    void rephase();                                         // do rephase.
    void printModel();                                      // print model when the result is SAT.
};