
'''The main task of this code file is using the Nelder Mead Method to find the optimal threshold T_i and T_o'''
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from simulator import server_simulator
from src.Config import *

## Experimental Settings
policy = "THRESHOLD"

# Choose which objective to optimise:
# "exact"  -> use Objective_Exact
# "little" -> use Objective_Little
OBJECTIVE_TYPE = "little"

# Common random numbers
OPTIMIZATION_SEEDS = list(range(100, 110))

# Cache for repeated rounded threshold combinations
objective_cache = {}

RESULT_DIR = "experiment_results/nelder_mead/"
os.makedirs(RESULT_DIR, exist_ok=True)

def threshold_constraints(turn_off_threshold, turn_on_threshold):
    """
    Check whether a threshold combination is feasible.
    T_i: turn-off threshold
    T_o: turn-on threshold
    Current policy interpretation:
    - T_i should be non-negative.
    - T_i cannot exceed number of servers.
    - T_o can be negative or non-negative.
      If T_o < 0, abs(T_o) is interpreted as a queue-length threshold.
      If T_o >= 0, T_o is interpreted as an idle-server threshold.
    """

    if turn_off_threshold < 0:
        return False

    if turn_off_threshold > NUM_SERVERS:
        return False

    if turn_on_threshold < -100 or turn_on_threshold > NUM_SERVERS:
        return False

    # If T_o >= 0 and T_i <= T_o, turn-on and turn-off rules may conflict.
    if turn_on_threshold >= 0 and turn_off_threshold <= turn_on_threshold:
        return False

    return True

def round_thresholds(decision_variable_x):
    """
    Nelder-Mead works in continuous space.
    The actual policy uses integer thresholds.

    decision_variable_x[0] corresponds to T_i.
    decision_variable_x[1] corresponds to T_o.
    """

    turn_off_threshold = int(round(decision_variable_x[0]))
    turn_on_threshold = int(round(decision_variable_x[1]))

    return turn_off_threshold, turn_on_threshold

def run_simulation(turn_off_threshold, turn_on_threshold, seed, phase="optimization"):
    (
        Average_System_Size,
        Utilization,
        Average_Power,
        Average_Waiting_Time,
        Average_Response_Time_Exact,
        Average_Response_Time_Little,
        ERP_Exact,
        ERP_Little,
        Objective_Exact,
        Objective_Little
    ) = server_simulator(
        Num_server=NUM_SERVERS,
        arrival_rate=ARRIVAL_RATE,
        service_rate=SERVICE_RATE,
        timesteps=SIMULATION_TIME,
        setup_time=SETUP_TIME,
        policy=policy,
        turn_off_threshold=turn_off_threshold,
        turn_on_threshold=turn_on_threshold,
        arrival_model=ARRIVAL_MODEL,
        arrival_scale_C=ARRIVAL_SCALE_C,
        arrival_alpha=ARRIVAL_ALPHA,
        arrival_amplitude=ARRIVAL_AMPLITUDE,
        seed=seed
    )

    if OBJECTIVE_TYPE == "exact":
        objective_value = Objective_Exact
    elif OBJECTIVE_TYPE == "little":
        objective_value = Objective_Little
    else:
        raise ValueError("OBJECTIVE_TYPE must be either 'exact' or 'little'.")

    record = {
        "phase": phase,
        "T_i": turn_off_threshold,
        "T_o": turn_on_threshold,
        "seed": seed,
        "average_system_size": Average_System_Size,
        "utilization": Utilization,
        "average_power": Average_Power,
        "average_waiting_time": Average_Waiting_Time,
        "average_response_time_exact": Average_Response_Time_Exact,
        "average_response_time_little": Average_Response_Time_Little,
        "ERP_exact": ERP_Exact,
        "ERP_little": ERP_Little,
        "objective_exact": Objective_Exact,
        "objective_little": Objective_Little,
        "selected_objective": objective_value
    }

    return record

def estimate_objective(turn_off_threshold, turn_on_threshold, seeds):

    records = []

    for seed in seeds:
        record = run_simulation(
            turn_off_threshold = turn_off_threshold,
            turn_on_threshold = turn_on_threshold,
            seed = seed
        )
        records.append(record)

    record_df = pd.DataFrame(records)
    mean_objective = float(record_df["selected_objective"].mean())

    return mean_objective, records

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
    mean_objective, _ = estimate_objective(turn_off_threshold=turn_off_threshold,
                                           turn_on_threshold=turn_on_threshold,
                                           seeds=OPTIMIZATION_SEEDS)

    if not np.isfinite(mean_objective):
        mean_objective = LARGE_PENALTY

    objective_cache[key] = mean_objective

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
            seed=seed
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

    print("\nFinal Best Result")
    print("Best T_i:", best_T_i)
    print("Best T_o:", best_T_o)
    print("Objective type:", OBJECTIVE_TYPE)
    print("Mean best objective:", best_mean_objective)

    print("\nObjective values by seed:")
    print(best_result_df[["seed", "selected_objective"]])
