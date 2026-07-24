
'''The main task of this code file is using the Nelder Mead Method to find the optimal threshold T_i and T_o'''
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

# Store evaluated points for later analysis
evaluation_records = []

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

    if turn_on_threshold < -20 or turn_on_threshold > NUM_SERVERS:
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

def run_simulation(T_i, T_o, seed, phase):
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
        turn_off_threshold=T_i,
        turn_on_threshold=T_o,
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
        "T_i": T_i,
        "T_o": T_o,
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

