import pandas as pd
from simulator import server_simulator
from src.Config import *

def threshold_constraints(turn_off_threshold,
                          turn_on_threshold,
                          num_servers=NUM_SERVERS,
                          turn_on_mode=None):
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

    if turn_off_threshold > num_servers:
        return False

    if turn_on_threshold < -num_servers or turn_on_threshold > num_servers:
        return False

    if turn_on_mode == "queue_based":
        if turn_on_threshold >= 0:
            return False

    elif turn_on_mode == "idle_based":
        if turn_on_threshold < 0:
            return False

    elif turn_on_mode is None:
        pass

    else:
        raise ValueError("turn_on_mode must be None, 'queue_based', or 'idle_based'.")

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

def run_simulation(turn_off_threshold,
                   turn_on_threshold,
                   seed,
                   policy="THRESHOLD",
                   response_method="little",
                   phase="optimization"):

    (
        Average_System_Size,
        Utilization,
        Average_Power,
        Average_Response_Time,
        ERP,
        Objective,
        Num_completed_users
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
        arrival_rho=ARRIVAL_RHO,
        response_method=response_method,
        seed=seed
    )

    record = {
        "phase": phase,
        "policy": policy,
        "response_method": response_method,

        # Experimental setting
        "num_servers": NUM_SERVERS,
        "simulation_time": SIMULATION_TIME,
        "arrival_model": ARRIVAL_MODEL,
        "arrival_rate_base": ARRIVAL_RATE,
        "arrival_scale_C": ARRIVAL_SCALE_C,
        "arrival_alpha": ARRIVAL_ALPHA,
        "arrival_amplitude": ARRIVAL_AMPLITUDE,
        "arrival_rho": ARRIVAL_RHO,
        "service_rate": SERVICE_RATE,
        "setup_time": SETUP_TIME,
        "response_time_weight": RESPONSE_TIME_WEIGHT,

        # Thresholds
        "T_i": turn_off_threshold,
        "T_o": turn_on_threshold,

        # Seed
        "seed": seed,

        # Performance
        "average_system_size": Average_System_Size,
        "utilization": Utilization,
        "average_power": Average_Power,
        "average_response_time": Average_Response_Time,
        "ERP": ERP,
        "selected_objective": Objective,
        "num_completed_users": Num_completed_users
    }

    return record

def estimate_objective(turn_off_threshold,
                       turn_on_threshold,
                       seeds,
                       evaluation_id,
                       unrounded_turn_off_threshold,
                       unrounded_turn_on_threshold,
                       policy="THRESHOLD",
                       response_method="little"):

    records = []

    for seed in seeds:
        record = run_simulation(
            turn_off_threshold=turn_off_threshold,
            turn_on_threshold=turn_on_threshold,
            seed=seed,
            policy=policy,
            response_method=response_method
        )

        record["evaluation_id"] = evaluation_id
        record["raw_T_i"] = unrounded_turn_off_threshold
        record["raw_T_o"] = unrounded_turn_on_threshold

        records.append(record)

    record_df = pd.DataFrame(records)
    mean_objective = float(record_df["selected_objective"].mean())

    return mean_objective, records