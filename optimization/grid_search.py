import os
import numpy as np
import pandas as pd
from scipy.stats import t

from src.Config import *

from src.optimization_utils import (
    threshold_constraints,
    estimate_objective,
    run_simulation
)

## Experimental settings

POLICY = "THRESHOLD"

# Choose which objective to optimise:
# "exact"  -> use Objective_Exact
# "little" -> use Objective_Little
OBJECTIVE_TYPE = "little"

# Common random numbers
GRID_SEARCH_SEEDS = list(range(100, 110))

# Result directory
RESULT_DIR = "experiment_results/grid_search/"
os.makedirs(RESULT_DIR, exist_ok=True)

def run_grid_search(turn_off_threshold_values, turn_on_threshold_values, seeds):

    all_records = []

    best_objective = np.inf
    best_turn_off_threshold = None
    best_turn_on_threshold = None

    evaluation_id = 0

    for turn_off_threshold in turn_off_threshold_values:
        for turn_on_threshold in turn_on_threshold_values:

            # Check if the threshold pair is feasible
            if not threshold_constraints(turn_off_threshold, turn_on_threshold):
                continue

            evaluation_id += 1

            print(f"Start grid search: Evaluation ID: {evaluation_id}")
            print(f"Threshold pair (T_i, T_o): ({turn_off_threshold, turn_on_threshold})")

            # Start evaluate the threshold pairs
            mean_objective, records = estimate_objective(turn_off_threshold=turn_off_threshold,
                                                         turn_on_threshold=turn_on_threshold,
                                                         seeds=seeds,
                                                         evaluation_id=evaluation_id,
                                                         unrounded_turn_off_threshold=turn_off_threshold,
                                                         unrounded_turn_on_threshold=turn_on_threshold,
                                                         policy=POLICY,
                                                         objective_type=OBJECTIVE_TYPE
                                                         )

            if not np.isfinite(mean_objective):
                mean_objective = LARGE_PENALTY

            # Update the best objective and its threshold pair
            if mean_objective < best_objective:
                best_objective = mean_objective
                best_turn_off_threshold = turn_off_threshold
                best_turn_on_threshold = turn_on_threshold

            # Recording the experimental results
            for record in records:
                record["phase"] = "grid_search"
                record["mean_objective_this_evaluation"] = mean_objective
                record["best_objective"] = best_objective
                record["best_T_i"] = best_turn_off_threshold
                record["best_T_o"] = best_turn_on_threshold
                record["num_seeds"] = len(seeds)

            all_records.extend(records)

            print(f"Finished grid search: Evaluation ID: {evaluation_id}")
            print(f"mean objective={mean_objective:.6f}")

    return all_records

def save_grid_search_by_seed(all_records):
    df = pd.DataFrame(all_records)

    output_path = os.path.join(
        RESULT_DIR,
        "grid_search_by_seed.csv"
    )

    df.to_csv(output_path, index=False)

    print("\nSaved grid search by-seed results to:", output_path)

    return df

if __name__ == "__main__":
    turn_off_threshold_values = range(95, NUM_SERVERS + 1, 1)
    turn_on_threshold_values = range(10, 31, 1)

    all_records = run_grid_search(turn_off_threshold_values, turn_on_threshold_values, GRID_SEARCH_SEEDS)
    grid_search_df = save_grid_search_by_seed(all_records)

    print("Grid search finished.")

    summary_df = (
        grid_search_df
        .groupby(["evaluation_id", "T_i", "T_o"])
        .agg(
            mean_objective=("selected_objective", "mean"),
            std_objective=("selected_objective", "std"),
            n=("selected_objective", "count"),
            mean_power=("average_power", "mean"),
            mean_response_time_exact=("average_response_time_exact", "mean"),
            mean_response_time_little=("average_response_time_little", "mean"),
        )
        .reset_index()
    )

    best_row = summary_df.loc[summary_df["mean_objective"].idxmin()]

    print("\nGrid Search Best Result")
    print("Best T_i:", int(best_row["T_i"]))
    print("Best T_o:", int(best_row["T_o"]))
    print("Objective type:", OBJECTIVE_TYPE)
    print("Best mean objective:", float(best_row["mean_objective"]))

