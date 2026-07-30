
'''The main task of this code file is using the Nelder Mead Method to find the optimal threshold T_i and T_o'''
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.Config import *

from src.optimization_utils import (
    threshold_constraints,
    round_thresholds,
    estimate_objective,
    run_simulation
)

## Experimental Settings
POLICY = "THRESHOLD"

# Choose which objective to optimise:
# "exact"  -> use Objective_Exact
# "little" -> use Objective_Little
OBJECTIVE_TYPE = "little"

# Common random numbers
OPTIMIZATION_SEEDS = list(range(100, 110))

# Cache for repeated rounded threshold combinations
objective_cache = {}

# Store the convergence process records
convergence_records = []

RESULT_DIR = "../experiment_results/nelder_mead/"
os.makedirs(RESULT_DIR, exist_ok=True)

def objective_nelder_mead(decision_variable_x):
    turn_off_threshold, turn_on_threshold = round_thresholds(decision_variable_x)

    # Check if the (T_i, T_o) obey the threshold constraints
    if not threshold_constraints(turn_off_threshold, turn_on_threshold):
        print(f"x={decision_variable_x}, rounded to T_i={turn_off_threshold}, T_o={turn_on_threshold}, ")
        print(f"infeasible, objective={LARGE_PENALTY}")
        return LARGE_PENALTY

    key = (turn_off_threshold, turn_on_threshold, OBJECTIVE_TYPE)

    # Check if the given (T_i, T_o) has been simulated in the past
    if key in objective_cache:
        cached_value = objective_cache[key]
        print(f"x={decision_variable_x}, rounded to T_i={turn_off_threshold}, T_o={turn_on_threshold}, ")
        print(f"cached objective={cached_value:.6f}")
        return cached_value

    # Estimate the objective value of the given (T_i, T_o) in different seeds
    evaluation_id = len(convergence_records) // len(OPTIMIZATION_SEEDS) + 1

    mean_objective, records = estimate_objective(
        turn_off_threshold=turn_off_threshold,
        turn_on_threshold=turn_on_threshold,
        seeds=OPTIMIZATION_SEEDS,
        evaluation_id=evaluation_id,
        unrounded_turn_off_threshold=decision_variable_x[0],
        unrounded_turn_on_threshold=decision_variable_x[1],
        policy=POLICY,
        objective_type=OBJECTIVE_TYPE
    )

    if not np.isfinite(mean_objective):
        mean_objective = LARGE_PENALTY

    objective_cache[key] = mean_objective

    # Current best after adding this new evaluation
    best_key = min(objective_cache, key=objective_cache.get)
    best_objective_so_far = objective_cache[best_key]
    best_turn_off_threshold_so_far, best_turn_on_threshold_so_far, _ = best_key

    # Add summary information to each seed-level record
    for record in records:
        record["mean_objective_this_evaluation"] = mean_objective
        record["best_objective_so_far"] = best_objective_so_far
        record["best_T_i_so_far"] = best_turn_off_threshold_so_far
        record["best_T_o_so_far"] = best_turn_on_threshold_so_far
        record["num_seeds"] = len(OPTIMIZATION_SEEDS)

    # Save every seed-level result for this evaluation
    convergence_records.extend(records)

    print( f"x={decision_variable_x}, rounded to T_i={turn_off_threshold}, T_o={turn_on_threshold}, ")
    print(f"mean objective={mean_objective:.6f}")
    return mean_objective

def run_nelder_mead(initial_point, initial_simplex):

    result = minimize(objective_nelder_mead,
                      x0=initial_point,
                      method='Nelder-Mead',
                      options={
                           "initial_simplex": initial_simplex,
                           "maxiter":80,
                           "xatol":0.1,
                           "fatol":0.1,
                           "disp":True
                       })

    unrounded_turn_off_threshold = result.x[0]
    unrounded_turn_on_threshold = result.x[1]

    turn_off_threshold, turn_on_threshold = round_thresholds(result.x)

    best_key = min(objective_cache, key=objective_cache.get)
    best_turn_off_threshold, best_turn_on_threshold, _ = best_key
    best_mean_objective = objective_cache[best_key]

    print("Nelder-Mead Finished")
    print("Raw optimizer result:", result.x)
    print("Rounded optimizer result: T_i =", turn_off_threshold, ", T_o =", turn_on_threshold)
    print("Best cached result: T_i =", best_turn_off_threshold, ", T_o =", best_turn_on_threshold)
    print("Best mean objective:", best_mean_objective)
    print("Success:", result.success)
    print("Message:", result.message)

    return best_turn_off_threshold, best_turn_on_threshold, best_mean_objective, result

def save_results_by_seed(best_turn_off_threshold, best_turn_on_threshold, best_mean_objective):
    """
    Save the best threshold pair and its value under each seed.
    """

    records = []

    for seed in OPTIMIZATION_SEEDS:
        record = run_simulation(
            turn_off_threshold=best_turn_off_threshold,
            turn_on_threshold=best_turn_on_threshold,
            seed=seed,
            policy=POLICY,
            objective_type=OBJECTIVE_TYPE
        )

        record["best_T_i"] = best_turn_off_threshold
        record["best_T_o"] = best_turn_on_threshold
        record["mean_best_objective"] = best_mean_objective
        record["objective_type"] = OBJECTIVE_TYPE

        records.append(record)

    df = pd.DataFrame(records)

    # Put the target columns first
    preferred_columns = [
        "best_T_i",
        "best_T_o",
        "objective_type",
        "mean_best_objective",
        "seed",
        "selected_objective",
        "average_power",
        "average_response_time_exact",
        "average_response_time_little",
        "objective_exact",
        "objective_little",
        "ERP_exact",
        "ERP_little",
        "average_system_size",
        "utilization",
        "average_waiting_time"
    ]

    df = df[preferred_columns]

    output_path = os.path.join(
        RESULT_DIR,
        "nelder_mead_best_result_by_seed.csv"
    )

    df.to_csv(output_path, index=False)

    print("\nSaved best result by seed to:", output_path)

    return df

def save_convergence_records():
    """
    Save the seed-level convergence process.

    One row = one evaluated threshold pair under one seed.
    Cached evaluations are not repeated.
    """

    df = pd.DataFrame(convergence_records)

    output_path = os.path.join(
        RESULT_DIR,
        "nelder_mead_convergence_by_seed.csv"
    )

    df.to_csv(output_path, index=False)

    print("\nSaved Nelder-Mead convergence records to:", output_path)

    return df


if __name__ == "__main__":

    initial_point = np.array([20.0, -10.0])

    initial_simplex = np.array([
        initial_point,
        initial_point + np.array([20.0, 0.0]),
        initial_point + np.array([0.0, -10.0])
    ])

    best_T_i, best_T_o, best_mean_objective, result = run_nelder_mead(
        initial_point,
        initial_simplex
    )

    best_result_df = save_results_by_seed(
        best_turn_off_threshold=best_T_i,
        best_turn_on_threshold=best_T_o,
        best_mean_objective=best_mean_objective
    )

    convergence_df = save_convergence_records()

    print("\nFinal Best Result")
    print("Best T_i:", best_T_i)
    print("Best T_o:", best_T_o)
    print("Objective type:", OBJECTIVE_TYPE)
    print("Mean best objective:", best_mean_objective)

    print("\nObjective values by seed:")
    print(best_result_df[["seed", "selected_objective"]])
