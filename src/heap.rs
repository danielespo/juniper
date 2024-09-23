// So I generally am not the best Rust programmer
// and CreuSAT Already does rust for their backend
// and is pretty good, am wondering if we want a SAT solver in C/C++ or in rust in the end of this
// I'll just KISS and do it in c++

// Note, smallest CoinsGrid_N45_C16.cnf on vanilla cadical (the command I gave)

// c --- [ statistics ] ---------------------------------------------------------
// c 
// c conflicts:                125875     13774.68    per second
// c decisions:              14399798   1575790.05    per second
// c fixed:                         8         0.04 %  of all variables
// c learned:                  125875       100.00 %  per conflict
// c learned_lits:            1004280       100.00 %  learned literals
// c minimized:                     0         0.00 %  learned literals
// c shrunken:                      0         0.00 %  learned literals
// c minishrunken:                  0         0.00 %  learned literals
// c otfs:                          0         0.00 %  of conflict
// c propagations:          100249625        10.97 M  per second
// c restarts:                   9964        12.63    interval
// c stabilizing:                   3        33.37 %  of conflicts
// c subsumed:                  10481         3.78 %  of all clauses
// c trail reuses:                  0         0.00 %  of incremental calls
// c 
// c seconds are measured in process time for solving
// c 
// c --- [ resources ] ----------------------------------------------------------
// c 
// c total process time since initialization:         9.20    seconds
// c total real time since initialization:            9.24    seconds
// c maximum resident set size of process:           50.59    MB
// c 
// c --- [ shutting down ] ------------------------------------------------------


// And then for default CaDiCal on the same SAT problem CoinsGrid_N45_C16.cnf


// c --- [ statistics ] ---------------------------------------------------------
// c 
// c chronological:             18694        42.05 %  of conflicts
// c conflicts:                 44454     20342.18    per second
// c decisions:                450037    205937.28    per second
// c eliminated:                  569         2.70 %  of all variables
// c fixed:                        30         0.14 %  of all variables
// c learned:                   39148        88.06 %  per conflict
// c learned_lits:             688653       100.00 %  learned literals
// c minimized:                     0         0.00 %  learned literals
// c shrunken:                  23058         3.35 %  learned literals
// c minishrunken:              28512         4.14 %  learned literals
// c otfs:                       6716        15.11 %  of conflict
// c propagations:            6223953         2.85 M  per second
// c reduced:                   17399        39.14 %  per conflict
// c rephased:                      8      5556.75    interval
// c restarts:                     20      2222.70    interval
// c substituted:                 567         2.69 %  of all variables
// c subsumed:                  30671         9.31 %  of all clauses
// c strengthened:             198347        60.19 %  of all clauses
// c ternary:                   17485         2.82 %  of resolved
// c trail reuses:                  0         0.00 %  of incremental calls
// c vivified:                   1232         0.37 %  of all clauses
// c walked:                        4     11113.50    interval
// c weakened:                   4779         3.73    average size
// c 
// c seconds are measured in process time for solving
// c 
// c --- [ resources ] ----------------------------------------------------------
// c 
// c total process time since initialization:         2.24    seconds
// c total real time since initialization:            2.26    seconds
// c maximum resident set size of process:           41.72    MB
// c 
// c --- [ shutting down ] ------------------------------------------------------

