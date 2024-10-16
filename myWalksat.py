import math
import walksat as wk
import wsatA1 as a1
import tinishWalksat as tw
import numpy as np

# Copyright Daniel Espinosa Gonzalez, Tinish Bhattacharya
# October 2024
# UCSB Strukov Lab

def get_stats(walksat_result, max_flips):
    # walksat_result = [[assignment, _try, _loop, flips], ...]
    flips_list = []
    success_flags = []

    for res in walksat_result:
        if res != "FAIL":
            # if not a fail then a success
            assignment, _Tries, _Flips, flips = res
            # if stops working then make _Flips in flips_list ..
            flips_list.append(flips)
            success_flags.append(1)
        else:
            # if fail then fail
            flips_list.append(max_flips)
            success_flags.append(0)

    flips_array = np.array(flips_list)
    success_array = np.array(success_flags)

    walksat_result_arr = np.column_stack((flips_array, success_array))

    prob_s = np.mean(success_array)
    success_ind = np.where(walksat_result_arr[:, 1] == 1)[0]

    # Average number of flips over successful runs
    if len(success_ind) > 0:
        avg_flips = np.mean(walksat_result_arr[success_ind, 0])
        std_flips = np.std(walksat_result_arr[success_ind, 0])
    else:
        avg_flips = 0
        std_flips = 0

    max_restarts = walksat_result_arr.shape[0]
    # Same as tinish code
    p_targ = 0.99
    sorted_arr = walksat_result_arr[walksat_result_arr[:, 0].argsort()]

    if prob_s >= p_targ:
        ind_tts = math.ceil(p_targ * max_restarts) - 1
        tts_99 = sorted_arr[ind_tts, 0]
    else:
        if prob_s > 0:
            tts_99 = sorted_arr[max_restarts - 1, 0] * (
                math.log(1 - p_targ, 10) / math.log(1 - prob_s, 10)
            )
        else:
            tts_99 = float('inf')  # If no success then no TTS99

    return avg_flips, prob_s, std_flips, tts_99

def main():

    filepath = "/home/dae/SatExperiments/juniper/uf50suiteSATLIB/uf5001.cnf"
    #"/home/dae/SatExperiments/juniper/uf50suiteSATLIB/uf5001.cnf"

    # These options are the same as tinishWalksat.py
    max_tries = 10 #same as tinish
    max_flips = 100000
    probability = 0.5

    try:
        num_vars, clauses = wk.read_dimacs(filepath)
    except Exception as e:
        print(f"Error reading CNF file: {e}")
        return
    
    walksat_result = []

    # "walksat", "coloringA1_heuristic0"
    # then change the heuristic number 0-3
    # heuristic_mode: 
    # 0 = greedy in colors from candidate variables to flip
    # 1 = random from candidate variables to flip
    # 2 = pick a random color from candidate variables to flip
    # 3 = always pick first candidate variable in candidate variables to flip

    mode = "walksat"

    for _ in range(10):
        if mode == "walksat":
            result = wk.walkSAT(clauses, max_tries, max_flips, probability)
            walksat_result.append(result)
        elif mode == "coloringA1_heuristic0":
            colors = a1.GenerateColors(clauses)
            max_loops = math.floor(max_flips / len(colors))
            result = a1.AlgorithmA1(clauses, colors, max_tries, max_loops, probability, 0)
            # result = [assignment, _try, _loop, flips]
            walksat_result.append(result)
        elif mode == "coloringA1_heuristic1":
            colors = a1.GenerateColors(clauses)
            max_loops = math.floor(max_flips / len(colors))
            result = a1.AlgorithmA1(clauses, colors, max_tries, max_loops, probability, 1)
            walksat_result.append(result)
        elif mode == "coloringA1_heuristic2":
            colors = a1.GenerateColors(clauses)
            max_loops = math.floor(max_flips / len(colors))
            result = a1.AlgorithmA1(clauses, colors, max_tries, max_loops, probability, 2)
            walksat_result.append(result)
        elif mode == "coloringA1_heuristic3":
            colors = a1.GenerateColors(clauses)
            max_loops = math.floor(max_flips / len(colors))
            result = a1.AlgorithmA1(clauses, colors, max_tries, max_loops, probability, 3)
            walksat_result.append(result)
    
    avg_flips, prob_s, std_flips, tts_99 = get_stats(walksat_result, max_flips)

    print("Average Number of Flips: ",avg_flips)
    print("Average Probability of Success: ",prob_s)
    print("Standard deviation of flips: ",std_flips)
    #print("TTS99: ",tts_99)

if __name__ == "__main__":
    main()