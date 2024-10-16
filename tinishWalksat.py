import numpy as np
import random
import sys
import math
from multiprocessing import Process
import concurrent.futures
import itertools
import argparse #added argument parsing

# Copyright Tinish Bhattacharya, 2024
# UCSB Strukov Lab
# Used with permission, Daniel Espinosa Gonzalez 2024

# Modified the parser to be directory-agnostic
# parser assumes cnf files have standard header as 0th row
# use the parser in wsatA1.py if you would prefer one which
# does not mind how much text there is :)

def sat_eval(var_val,clause_mat):
    
    nvar = len(var_val)
    nclause = clause_mat.shape[0]
    clause_stat_mat = np.zeros((nclause,1),dtype=bool)
    clause_f_mat = np.zeros((nclause,1))
    sat_result = 1
    
    for i in range(nclause):
        
        var_ind = np.where(clause_mat[i,]!=0)[0]
        clause_stat_mat[i], clause_f_mat[i] = clause_eval(var_val[var_ind],clause_mat[i,var_ind])
        sat_result = sat_result and clause_stat_mat[i]
        
    return clause_stat_mat, sat_result, clause_f_mat
        
def clause_eval(var_val,var_sign):
    
    nvar = len(var_val)
    clause_sum = 0
    clause_f_count = 0
    for i in range(nvar):
        if var_sign[i]==1:
            clause_sum = clause_sum or var_val[i]
            clause_f_count = clause_f_count + (1 if var_val[i] else 0)
        elif var_sign[i]==-1:
            clause_sum = clause_sum or ~var_val[i]
            clause_f_count = clause_f_count + (1 if ~var_val[i] else 0)
        else:
            clause_sum = clause_sum
            
    return clause_sum, clause_f_count

def rand_init(nvar):
    
    var_val_mat = np.random.rand(nvar,1)
    var_val_mat = [1 if (i>0.5) else 0 for i in var_val_mat]
    var_val_mat = np.array(var_val_mat,dtype=bool)
    
    return var_val_mat

def number_of_sat_clause(clause_stat_mat):
    
    return len(np.where(clause_stat_mat==1)[0])

def G_heuristics(p,var_val,clause_mat,clause_f_mat,clause_ind,clause_stat_mat):
    
    nvar = len(var_val)
    nclause = clause_mat.shape[0]
    var_list = np.where(clause_mat[clause_ind,]!=0)[0]
    num_sat_clause_init = number_of_sat_clause(clause_stat_mat)
    clause_stat_change_mat = np.zeros((len(var_list),1))
    var_counter = 0
    xx = random.random()
    if xx <= p:
        var_candidate = var_list
    else:
        for i in var_list:

            clause_stat_change_mat[var_counter], _ = delta_E(i,clause_ind,clause_mat,clause_stat_mat,clause_f_mat)
            var_counter = var_counter + 1

        var_candidate = var_list[np.where(clause_stat_change_mat>0)[0]]
        if len(var_candidate)==0:
            var_candidate = var_list
        #else:
         #   var_candidate = var_list[np.where(clause_stat_change_mat==np.max(clause_stat_change_mat))[0]]
    
    chosen_var = np.random.choice(var_candidate,size=1)[0]
    var_val[chosen_var] = ~var_val[chosen_var]
    clause_stat_mat_new, sat_result_new, clause_f_mat_new = get_clause_stat(chosen_var,clause_ind,clause_mat,clause_stat_mat,clause_f_mat)
    
    return var_val, clause_stat_mat_new, sat_result_new, clause_f_mat_new

def Shonning_heuristics(p,var_val,clause_mat,clause_f_mat,clause_ind,clause_stat_mat):
    
    nvar = len(var_val)
    nclause = clause_mat.shape[0]
    var_list = np.where(clause_mat[clause_ind,]!=0)[0]    
    chosen_var = np.random.choice(var_list,size=1)[0]
    var_val[chosen_var] = ~var_val[chosen_var]
    clause_stat_mat_new, sat_result_new, clause_f_mat_new = get_clause_stat(chosen_var,clause_ind,clause_mat,clause_stat_mat,clause_f_mat)
    
    return var_val, clause_stat_mat_new, sat_result_new, clause_f_mat_new

def get_stats(walksat_result):
    
    prob_s = np.mean(walksat_result[:,1])
    success_ind = np.where(walksat_result[:,1]==1)[0]
    avg_flips = np.mean(walksat_result[success_ind,0])
    std_flips = np.std(walksat_result[success_ind,0])
    
    max_restarts = np.shape(walksat_result)[0]
    p_targ = 0.99
    sorted_arr = walksat_result[walksat_result[:,0].argsort()]
    if prob_s>=p_targ:
        ind_tts = math.ceil(p_targ*max_restarts)-1
        tts_99 = sorted_arr[ind_tts,0]
    else:
        tts_99 = sorted_arr[max_restarts-1,0]*(math.log((1-p_targ),10)/math.log((1-prob_s),10))
    
    return avg_flips, prob_s, std_flips, tts_99

def SKC_heuristics(p,var_val,clause_mat,clause_f_mat,clause_ind,clause_stat_mat):
    
    nvar = len(var_val)
    nclause = clause_mat.shape[0]
    var_list = np.where(clause_mat[clause_ind,]!=0)[0]
    num_sat_clause_init = number_of_sat_clause(clause_stat_mat)
    clause_stat_change_mat = np.zeros((len(var_list),1))
    var_break_value_mat = np.zeros((len(var_list),1))
    var_counter = 0
    xx = random.random()
   
    for i in var_list:

        clause_stat_change_mat[var_counter], var_break_value_mat[var_counter] = delta_E(i,clause_ind,clause_mat,clause_stat_mat,clause_f_mat)
        var_counter = var_counter + 1

    break_value_zero_ind = np.where(var_break_value_mat==0)[0]
    
    if len(break_value_zero_ind)!=0:
        chosen_var = var_list[np.random.choice(break_value_zero_ind,size=1)[0]]
    elif xx <= p:
        chosen_var = np.random.choice(var_list[np.where(clause_stat_change_mat==np.max(clause_stat_change_mat))[0]],size=1)[0]
    else:
        chosen_var = np.random.choice(var_list,size=1)[0]
    
    var_val[chosen_var] = ~var_val[chosen_var]
    clause_stat_mat_new, sat_result_new, clause_f_mat_new = get_clause_stat(chosen_var,clause_ind,clause_mat,clause_stat_mat,clause_f_mat)
    
    return var_val, clause_stat_mat_new, sat_result_new, clause_f_mat_new
    
def delta_E(i,clause_ind,clause_mat,clause_stat_mat,clause_f_mat):
    
    var_sign = clause_mat[clause_ind,i]
    A_sign = var_sign
    B_sign = -1*var_sign
    A_clause_ind1 = np.where(clause_mat[:,i]==A_sign)[0]
    A = len(np.where(clause_stat_mat[A_clause_ind1]==0)[0])
    B_clause_ind1 = np.where(clause_mat[:,i]==B_sign)[0]
    B = len(np.where(clause_f_mat[B_clause_ind1]==1)[0])
    gain = A-B
    
    return gain, B
    
def get_clause_stat(i,clause_ind,clause_mat,clause_stat_mat,clause_f_mat):
    var_sign = clause_mat[clause_ind,i]
    A_sign = var_sign
    B_sign = -var_sign
    A_clause_ind = np.where(clause_mat[:,i]==A_sign)[0]
    A_clause_ind_pa = np.intersect1d(A_clause_ind,np.where(clause_stat_mat==0)[0])
    A_clause_ind_pb = np.intersect1d(A_clause_ind,np.where(clause_stat_mat==1)[0])
    B_clause_ind = np.where(clause_mat[:,i]==B_sign)[0]
    B_clause_ind_pa = np.intersect1d(B_clause_ind,np.where(clause_f_mat==1)[0])
    B_clause_ind_pb = np.intersect1d(B_clause_ind,np.where(clause_f_mat!=1)[0])
    clause_stat_mat_new = np.array(clause_stat_mat)
    clause_f_mat_new = np.array(clause_f_mat)
    clause_stat_mat_new[A_clause_ind_pa] = True
    clause_stat_mat_new[B_clause_ind_pa] = False
    clause_f_mat_new[A_clause_ind] = clause_f_mat_new[A_clause_ind] + 1
    clause_f_mat_new[B_clause_ind] = clause_f_mat_new[B_clause_ind] - 1
    sat_result_new = 1 if (np.sum(clause_stat_mat_new)==len(clause_stat_mat_new)) else 0
    
    return clause_stat_mat_new, sat_result_new, clause_f_mat_new

def WalkSat_Solver(p,clause_mat,max_restarts,max_flips,heuristics):
    
    nvar = clause_mat.shape[1]
    nclause = clause_mat.shape[0]
    
    walksat_result = np.zeros((max_restarts,2))

    for r in range(max_restarts):

        var_val_mat = rand_init(nvar)
        clause_stat_mat, sat_result, clause_f_mat = sat_eval(var_val_mat,clause_mat)
        
        for f in range(max_flips):

            if sat_result == 1:
                walksat_result[r,0] = f
                walksat_result[r,1] = 1
                break

            unsat_clause_list = np.where(clause_stat_mat==0)[0]
            chosen_clause_ind = np.random.choice(unsat_clause_list,size=1)[0]
            if heuristics == "SKC":
                var_val_mat, clause_stat_mat, sat_result, clause_f_mat = SKC_heuristics(p,var_val_mat,clause_mat,clause_f_mat,chosen_clause_ind,clause_stat_mat)
            elif heuristics == "G":
                var_val_mat, clause_stat_mat, sat_result, clause_f_mat = G_heuristics(p,var_val_mat,clause_mat,clause_f_mat,chosen_clause_ind,clause_stat_mat)
            else:
                var_val_mat, clause_stat_mat, sat_result, clause_f_mat = Shonning_heuristics(p,var_val_mat,clause_mat,clause_f_mat,chosen_clause_ind,clause_stat_mat)

        if walksat_result[r,1] == 0:
            clause_stat_mat, sat_result, clause_f_mat = sat_eval(var_val_mat,clause_mat)
            walksat_result[r,0] = max_flips
            if sat_result == 1:
                walksat_result[r,1] = 1  

    avg_flips, prob_s, std_flips, tts_99 = get_stats(walksat_result)
    
    return avg_flips, prob_s, std_flips, tts_99

def WalkSat_Solver_rld(p,clause_mat,max_restarts,max_flips,heuristics):
    
    nvar = clause_mat.shape[1]
    nclause = clause_mat.shape[0]
    
    walksat_result = np.zeros((max_restarts,2))

    for r in range(max_restarts):

        var_val_mat = rand_init(nvar)
        clause_stat_mat, sat_result, clause_f_mat = sat_eval(var_val_mat,clause_mat)
        
        for f in range(max_flips):

            if sat_result == 1:
                walksat_result[r,0] = f
                walksat_result[r,1] = 1
                break

            unsat_clause_list = np.where(clause_stat_mat==0)[0]
            chosen_clause_ind = np.random.choice(unsat_clause_list,size=1)[0]
            if heuristics == "SKC":
                var_val_mat, clause_stat_mat, sat_result, clause_f_mat = SKC_heuristics(p,var_val_mat,clause_mat,clause_f_mat,chosen_clause_ind,clause_stat_mat)
            elif heuristics == "G":
                var_val_mat, clause_stat_mat, sat_result, clause_f_mat = G_heuristics(p,var_val_mat,clause_mat,clause_f_mat,chosen_clause_ind,clause_stat_mat)
            else:
                var_val_mat, clause_stat_mat, sat_result, clause_f_mat = Shonning_heuristics(p,var_val_mat,clause_mat,clause_f_mat,chosen_clause_ind,clause_stat_mat)

        if walksat_result[r,1] == 0:
            clause_stat_mat, sat_result, clause_f_mat = sat_eval(var_val_mat,clause_mat)
            walksat_result[r,0] = max_flips
            if sat_result == 1:
                walksat_result[r,1] = 1  

    avg_flips, prob_s, std_flips, tts_99 = get_stats(walksat_result)
    
    return walksat_result, tts_99   

def WalkSat_Solver_full(nvar,i,params):
    
    max_restarts = params['max_res']
    max_flips = params['max_flip']
    p = params['p']
    var_heur = params['var_heur']
    clause_mat = get_kSATprob(nvar,i)
    avg_flips, prob_s, std_flips, tts_99 = WalkSat_Solver(p,clause_mat,max_restarts,max_flips,var_heur)
    return avg_flips, prob_s, std_flips, i, tts_99
    
def WalkSat_Solver_full_rld(nvar,i,params):
    
    max_restarts = params['max_res']
    max_flips = params['max_flip']
    p = params['p']
    var_heur = params['var_heur']
    clause_mat = get_kSATprob(nvar,i)
    walksat_result, tts_99 = WalkSat_Solver_rld(p,clause_mat,max_restarts,max_flips,var_heur)
    return walksat_result, i, tts_99
    
def arrange_flips(walksat_result):
    max_restarts = np.shape(walksat_result)[0]
    final_arr = np.zeros((1,max_restarts))
    for i in range(max_restarts):
        if walksat_result[i,1]==1:
            final_arr[0,i] = walksat_result[i,0]
            
    final_arr = final_arr[0,final_arr[0,].argsort()]
    
    return final_arr

def get_kSATprob(file_addr="/home/dae/SatExperiments/juniper/TestFolderCNF/abitbigger.cnf"):
    """
    Ex.    p cnf 3 6
    startfile = p 
    cnf = filetype
    3 = nvar
    6 = nclauses

    instance = filename i.g. "coloring.cnf"

    Inputs:

    Outputs:
    A numpy matrix of clauses , each submatrix being k long, where k-SAT eg 3SAT, 4SAT etc.

    """
    # File address, change back to be able to run through dirs
    #file_addr = './SATLIB_Data/'+dir_name+'/'+instance_name+'.cnf'
    file_addr = "/home/dae/SatExperiments/juniper/uf50suiteSATLIB/uf5001.cnf"
    #"/home/dae/SatExperiments/juniper/uf50suiteSATLIB/uf50016.cnf"

    with open(file_addr) as f:
        lines = f.readlines()
    f.close()

    header = lines[0]
    split_header = header.split(' ')
    k = 3 # order of SAT, say 3SAT, 4SAT, etc.
    nvar = int(split_header[2])
    nclause = int(split_header[3])
    line_start = 1
    line_end = line_start + nclause
    clause_mat = np.zeros((nclause,nvar))

    for i in np.arange(line_start,line_end,1):
        cl_temp = np.array([int(kk) for kk in lines[i].split()])
        clause_mat[i-line_start,np.absolute(cl_temp[0:k])-1] = np.sign(cl_temp[0:k])
        
    return clause_mat

def main():
    # NOTE:
    # This runs 10 times to get the statistics (as indicated by max_restarts)
    # instead of doing shot by shot and then aggregating. 

    nvar = 3
    instance = 10
    max_restarts = 10
    max_flips = 100000
    p = 0.5

    clause_mat = get_kSATprob()
    avg_flips, prob_s, std_flips, tts_99 = WalkSat_Solver(p,clause_mat,max_restarts,max_flips,"SKC")
    print("Average Number of Flips: ",avg_flips)
    print("Average Probability of Success: ",prob_s)
    print("Standard deviation of flips: ",std_flips)

if __name__ == "__main__":
    main()