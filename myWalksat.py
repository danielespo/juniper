import math
import walksat as wk
import wsatA1 as a1
import tinishWalksat as tw
import numpy as np
from multiprocessing import Pool

# Copyright Daniel Espinosa Gonzalez, Tinish Bhattacharya
# October 2024
# UCSB Strukov Lab

def get_stats(walksat_result, max_flips):
    # walksat_result = [[assignment, _try, _loop, flips], ...]
    flips_list = []
    loops_list = []
    tries_list = []
    success_flags = []
    max_tries = 10 #change me
    for res in walksat_result:
        if res != "FAIL":
            # if not a fail then a success
            assignment, _Tries, _Loops, flips = res
            # Flips here has to be flips and not _flips because
            # _flips is actually _loops in wsatA1 and does not correspond
            # to actual flips per the algorithm. In walksat.py the flips
            # variable is duplicated for this reason.
            # Renamed to "_Loops" to remove confusion
            flips_list.append(flips)
            loops_list.append(_Loops)
            tries_list.append(_Tries)
            success_flags.append(1)
        else:
            # if fail then fail
            flips_list.append(max_flips)
            loops_list.append(max_flips)
            tries_list.append(max_tries)
            success_flags.append(0)

    flips_array = np.array(flips_list)
    loops_array = np.array(loops_list)
    tries_array = np.array(tries_list)
    success_array = np.array(success_flags)

    walksat_result_arr = np.column_stack((flips_array, loops_list, tries_list, success_array))

    prob_s = np.mean(success_array)
    
    success_ind = np.where(success_array == 1)[0]
    # Average number of flips over successful runs
    if len(success_ind) > 0:
        avg_flips = np.mean(walksat_result_arr[success_ind, 0])
        std_flips = np.std(walksat_result_arr[success_ind, 0])

        avg_loops = np.mean(walksat_result_arr[success_ind, 1])
        std_loops = np.std(walksat_result_arr[success_ind, 1])

        avg_tries = np.mean(walksat_result_arr[success_ind, 2])
        std_tries = np.std(walksat_result_arr[success_ind, 2])
    else:
        avg_flips = 0
        std_flips = 0
        avg_loops = 0
        std_loops = 0
        avg_tries = 0
        std_tries = 0


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

    return avg_flips, prob_s, std_flips, tts_99, avg_loops, std_loops, avg_tries, std_tries


def API(filepath, max_tries, max_flips, probability, mode):

    #filepath = "/home/dae/SatExperiments/juniper/uf50suiteSATLIB/uf5001.cnf"
    #"/home/dae/SatExperiments/juniper/uf50suiteSATLIB/uf5001.cnf"

    # These options are the same as tinishWalksat.py
    #max_tries = 10 #same as tinish
    #max_flips = 100000
    #probability = 0.5

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

    # mode = "coloringA1_heuristic0"

    walksat_result = []

    with Pool(processes=10) as pool:
        result_objects = []

        # The range(10) means 10 independent tries of max_tries and max_flips each
        # To get the CDF plot, must increase max_flips slowly and then get the data from there
        # Adding a mode called "CDF" to do this, in a second.

        if mode == "walksat":
            for _ in range(10):
                res = pool.apply_async(wk.walkSAT, args=(clauses, max_tries, max_flips, probability))
                result_objects.append(res)

        elif mode == "coloringA1_heuristic0":
            colors = a1.GenerateColors(clauses)
            max_loops = math.floor(max_flips / len(colors))
            for _ in range(10):
                res = pool.apply_async(a1.AlgorithmA1, args=(clauses, colors, max_tries, max_loops, probability, 0)) # assignment[1:], _try, _loop, flips
                result_objects.append(res)

        elif mode == "coloringA1_heuristic1":
            colors = a1.GenerateColors(clauses)
            max_loops = math.floor(max_flips / len(colors))
            for _ in range(10):
                res = pool.apply_async(a1.AlgorithmA1, args=(clauses, colors, max_tries, max_loops, probability, 1))
                result_objects.append(res)

        elif mode == "coloringA1_heuristic2":
            colors = a1.GenerateColors(clauses)
            max_loops = math.floor(max_flips / len(colors))
            for _ in range(10):
                res = pool.apply_async(a1.AlgorithmA1, args=(clauses, colors, max_tries, max_loops, probability, 2))
                result_objects.append(res)

        elif mode == "coloringA1_heuristic3":
            colors = a1.GenerateColors(clauses)
            max_loops = math.floor(max_flips / len(colors))
            for _ in range(10):
                res = pool.apply_async(a1.AlgorithmA1, args=(clauses, colors, max_tries, max_loops, probability, 3))
                result_objects.append(res)

        for res in result_objects:
            walksat_result.append(res.get())
    pool.close()
    pool.join()
    
    avg_flips, prob_s, std_flips, tts_99, avg_loops, std_loops, avg_tries, std_tries = get_stats(walksat_result, max_flips)

    # Now returns also the avg_iterations (loops) and avg_tries

    # Output of a1: assignment[1:], _try, _loop, flips
    return avg_flips, prob_s, std_flips, tts_99, avg_loops, std_loops, avg_tries, std_tries


def API_CNF(filepath, max_tries, max_flips, probability, mode):
    """
    API to return the CNF for a given problem and a given algorithm, by iteratively increasing
    the number of flips allowed until a certain probability of convergence guaranteed

    Ex. 20% success probability with 100 flips
    40% success probablity with 500 flips
    100% success probability with 2000 flips

    etc.
    """
    return

def main():

    filepath = "/home/dae/SatExperiments/juniper/BenchmarkSubsetPaper/uf5002.cnf"
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

    mode = "coloringA1_heuristic2"

    walksat_result = []

    with Pool(processes=10) as pool:
        result_objects = []

        if mode == "walksat":
            for _ in range(10):
                res = pool.apply_async(wk.walkSAT, args=(clauses, max_tries, max_flips, probability))
                result_objects.append(res)

        elif mode == "coloringA1_heuristic0": # Pick candidate variables to flip with most color representation
            colors = a1.GenerateColors(clauses)

            # 11/11/2024 2:36pm
            # The initial idea in adding loops to the algorithm
            # was to limit the number of heuristic iterations by the number of colors
            # say, there are 10 colors in the preprocessing graph, and we have 
            max_loops = math.floor(max_flips / len(colors))
            for _ in range(10):
                res = pool.apply_async(a1.AlgorithmA1, args=(clauses, colors, max_tries, max_loops, probability, 0))
                result_objects.append(res)

        elif mode == "coloringA1_heuristic1": # Randomly pick a variable from the candidate variables to flip
            colors = a1.GenerateColors(clauses)
            max_loops = math.floor(max_flips / len(colors))
            for _ in range(10):
                res = pool.apply_async(a1.AlgorithmA1, args=(clauses, colors, max_tries, max_loops, probability, 1))
                result_objects.append(res)

        elif mode == "coloringA1_heuristic2": # Randomly pick a color from candidate variables to flip, flip all candidate variables of that color
            colors = a1.GenerateColors(clauses)
            max_loops = math.floor(max_flips / len(colors))
            for _ in range(10):
                res = pool.apply_async(a1.AlgorithmA1, args=(clauses, colors, max_tries, max_loops, probability, 2))
                result_objects.append(res)

        elif mode == "coloringA1_heuristic3": # Always pick the first candidate variable and flip it
            colors = a1.GenerateColors(clauses)
            max_loops = math.floor(max_flips / len(colors))
            for _ in range(10):
                res = pool.apply_async(a1.AlgorithmA1, args=(clauses, colors, max_tries, max_loops, probability, 3))
                result_objects.append(res)

        for res in result_objects:
            walksat_result.append(res.get())
    pool.close()
    pool.join()
    avg_flips, prob_s, std_flips, tts_99, avg_loops, std_loops, avg_tries, std_tries = get_stats(walksat_result, max_flips)

    print("Average Number of Flips: ",avg_flips)
    print("Average Probability of Success: ",prob_s)
    print("Standard deviation of flips: ",std_flips)
    print("TTS99: ", tts_99)
    print("Average Number of Iterations: ", avg_loops)
    print("Standard deviation of Iterations: ",std_loops)
    print("Average Number of Tries: ", avg_tries)
    print("Standard deviation of Tries: ",std_tries)

if __name__ == "__main__":
    main()