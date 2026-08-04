
'''The main task of this code file is using the Nelder Mead Method to find the optimal threshold T_i and T_o'''
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.Config import *

from src.experiment_utils import build_experiment_tag
from src.optimization_utils import (
    threshold_constraints,
    round_thresholds,
    estimate_objective,
    run_simulation
)
from pathlib import Path

## Experimental Settings
POLICY = "THRESHOLD"

# Choose which objective to optimise:
# "exact"  -> use Objective_Exact
# "little" -> use Objective_Little
RESPONSE_METHOD = "little"

TURN_ON_MODE = "queue_based"

# Common random numbers
OPTIMIZATION_SEEDS = list(range(100, 110))

# Cache for repeated rounded threshold combinations
objective_cache = {}

# Store the convergence process records
convergence_records = []

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_DIR = __import__("pathlib").Path(r"C:\Users\caoxb\PycharmProjects\ATM_simulation\experiment_results\nelder_mead")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

def get_current_experiment_tag():
    return build_experiment_tag(
        num_servers=NUM_SERVERS,
        arrival_model=ARRIVAL_MODEL,
        arrival_scale_C=ARRIVAL_SCALE_C,
        arrival_alpha=ARRIVAL_ALPHA,
        setup_time=SETUP_TIME,
        simulation_time=SIMULATION_TIME,
        turn_on_mode=TURN_ON_MODE,
        response_method=RESPONSE_METHOD,
        num_seeds=len(OPTIMIZATION_SEEDS)
    )

def build_initial_point(num_servers, turn_on_mode):
    """
    Automatically construct a feasible initial point and initial simplex
    for Nelder-Mead.

    decision variable:
        x[0] = T_i
        x[1] = T_o
    """

    if turn_on_mode == "queue_based":
        initial_T_i = int(round(0.85 * num_servers))
        initial_T_i = max(0, min(initial_T_i, num_servers))

        initial_queue_threshold = int(round(0.50 * num_servers))
        initial_queue_threshold = max(1, min(initial_queue_threshold, num_servers))

        initial_T_o = -initial_queue_threshold

        step_T_i = max(5, int(round(0.10 * num_servers)))
        step_queue_threshold = max(5, int(round(0.25 * num_servers)))

        simplex_T_i = min(initial_T_i + step_T_i, num_servers)
        simplex_queue_threshold = min(
            initial_queue_threshold + step_queue_threshold,
            num_servers
        )

        initial_simplex = np.array([
            [initial_T_i, initial_T_o],
            [simplex_T_i, initial_T_o],
            [initial_T_i, -simplex_queue_threshold]
        ])

        return np.array([initial_T_i, initial_T_o]), initial_simplex

    elif turn_on_mode == "idle_based":
        # Need T_i > T_o >= 0
        initial_T_i = int(round(0.8 * num_servers))
        initial_T_o = int(round(0.2 * num_servers))

        # Make sure T_i > T_o
        if initial_T_i <= initial_T_o:
            initial_T_i = min(num_servers, initial_T_o + 1)

        step_T_i = max(1, int(round(0.1 * num_servers)))
        step_T_o = max(1, int(round(0.1 * num_servers)))

        point_1 = np.array([initial_T_i, initial_T_o], dtype=float)

        point_2 = np.array([
            min(num_servers, initial_T_i + step_T_i),
            initial_T_o
        ], dtype=float)

        point_3 = np.array([
            initial_T_i,
            min(initial_T_i - 1, initial_T_o + step_T_o)
        ], dtype=float)

    else:
        raise ValueError("turn_on_mode must be either 'queue_based' or 'idle_based'.")

    initial_simplex = np.array([
        point_1,
        point_2,
        point_3
    ])

    return point_1, initial_simplex

def objective_nelder_mead(decision_variable_x):
    turn_off_threshold, turn_on_threshold = round_thresholds(decision_variable_x)

    # Check if the (T_i, T_o) obey the threshold constraints
    if not threshold_constraints(turn_off_threshold, turn_on_threshold, NUM_SERVERS, TURN_ON_MODE):
        print(f"x={decision_variable_x}, rounded to T_i={turn_off_threshold}, T_o={turn_on_threshold}, ")
        print(f"infeasible, objective={LARGE_PENALTY}")
        return LARGE_PENALTY

    key = (
        turn_off_threshold,
        turn_on_threshold,
        RESPONSE_METHOD,
        TURN_ON_MODE
    )
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
        response_method=RESPONSE_METHOD
    )

    if not np.isfinite(mean_objective):
        mean_objective = LARGE_PENALTY

    objective_cache[key] = mean_objective

    # Current best after adding this new evaluation
    best_key = min(objective_cache, key=objective_cache.get)
    best_objective_so_far = objective_cache[best_key]
    best_turn_off_threshold_so_far, best_turn_on_threshold_so_far, _, _ = best_key

    # Add summary information to each seed-level record
    for record in records:
        record["turn_on_mode"] = TURN_ON_MODE
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
    best_turn_off_threshold, best_turn_on_threshold, _, _ = best_key
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
            response_method=RESPONSE_METHOD,
            phase="best_result"
        )

        record["best_T_i"] = best_turn_off_threshold
        record["best_T_o"] = best_turn_on_threshold
        record["mean_best_objective"] = best_mean_objective
        record["response_method"] = RESPONSE_METHOD
        record["turn_on_mode"] = TURN_ON_MODE

        records.append(record)

    df = pd.DataFrame(records)

    preferred_columns = [
        "num_servers",
        "simulation_time",
        "arrival_model",
        "arrival_rate_base",
        "arrival_scale_C",
        "arrival_alpha",
        "arrival_amplitude",
        "service_rate",
        "setup_time",
        "response_time_weight",

        "best_T_i",
        "best_T_o",
        "turn_on_mode",
        "response_method",
        "mean_best_objective",

        "seed",
        "selected_objective",
        "average_power",
        "average_response_time",
        "ERP",
        "average_system_size",
        "utilization",
        "num_completed_users"
    ]

    df = df[preferred_columns]

    experiment_tag = get_current_experiment_tag()

    output_path = RESULT_DIR / f"nelder_mead_best_result_by_seed_{experiment_tag}.csv"

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

    experiment_tag = get_current_experiment_tag()

    output_path = RESULT_DIR / f"nelder_mead_convergence_by_seed_{experiment_tag}.csv"

    df.to_csv(output_path, index=False)

    print("\nSaved Nelder-Mead convergence records to:", output_path)

    return df


if __name__ == "__main__":
    initial_point, initial_simplex = build_initial_point(
        num_servers=NUM_SERVERS,
        turn_on_mode=TURN_ON_MODE
    )

    print("Initial point:", initial_point)
    print("Initial simplex:")
    print(initial_simplex)

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
    print("Response Method:", RESPONSE_METHOD)
    print("Mean best objective:", best_mean_objective)

    print("\nObjective values by seed:")
    print(best_result_df[["seed", "selected_objective"]])
