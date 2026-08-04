import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.Config import *
from src.optimization_utils import (
    threshold_constraints,
    estimate_objective
)


# ============================================================
# Local grid validation settings
# ============================================================

POLICY = "THRESHOLD"
RESPONSE_METHOD = "little"
TURN_ON_MODE = "queue_based"

GRID_SEARCH_SEEDS = list(range(100, 110))

# This file is only used to validate the suspicious point:
# n = 200, queue_based, T_i = 160, T_o = -26.
TURN_OFF_THRESHOLD_VALUES = [170, 175, 180, 185, 190, 195]
TURN_ON_THRESHOLD_VALUES = [-100, -120, -140, -160, -180, -196, -200]


# ============================================================
# Result directory
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "experiment_results" / "grid_search"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def run_grid_search(turn_off_threshold_values,
                    turn_on_threshold_values,
                    seeds):
    """
    Run a local grid search for threshold validation.

    One threshold pair is evaluated under multiple seeds.
    The objective value used for comparison is the mean objective
    over all seeds.
    """

    all_records = []

    best_objective = np.inf
    best_turn_off_threshold = None
    best_turn_on_threshold = None

    evaluation_id = 0

    for turn_off_threshold in turn_off_threshold_values:
        for turn_on_threshold in turn_on_threshold_values:

            # Check feasibility under queue_based mode
            if not threshold_constraints(
                turn_off_threshold,
                turn_on_threshold,
                num_servers=NUM_SERVERS,
                turn_on_mode=TURN_ON_MODE
            ):
                print(
                    f"Skip infeasible pair: "
                    f"T_i={turn_off_threshold}, T_o={turn_on_threshold}"
                )
                continue

            evaluation_id += 1

            print(f"Start grid search: Evaluation ID: {evaluation_id}")
            print(f"Threshold pair (T_i, T_o): ({turn_off_threshold, turn_on_threshold})")

            mean_objective, records = estimate_objective(
                turn_off_threshold=turn_off_threshold,
                turn_on_threshold=turn_on_threshold,
                seeds=seeds,
                evaluation_id=evaluation_id,
                unrounded_turn_off_threshold=turn_off_threshold,
                unrounded_turn_on_threshold=turn_on_threshold,
                policy=POLICY,
                response_method=RESPONSE_METHOD
            )

            if not np.isfinite(mean_objective):
                mean_objective = LARGE_PENALTY

            if mean_objective < best_objective:
                best_objective = mean_objective
                best_turn_off_threshold = turn_off_threshold
                best_turn_on_threshold = turn_on_threshold

            for record in records:
                record["phase"] = "local_grid_validation"
                record["turn_on_mode"] = TURN_ON_MODE
                record["mean_objective_this_evaluation"] = mean_objective
                record["best_objective_so_far"] = best_objective
                record["best_T_i_so_far"] = best_turn_off_threshold
                record["best_T_o_so_far"] = best_turn_on_threshold
                record["num_seeds"] = len(seeds)

            all_records.extend(records)

            print(f"Mean objective = {mean_objective:.6f}")
            print(
                f"Best so far: T_i={best_turn_off_threshold}, "
                f"T_o={best_turn_on_threshold}, "
                f"objective={best_objective:.6f}"
            )

    return all_records


def save_grid_search_results(all_records):
    """
    Save both seed-level results and summary results.
    """

    by_seed_df = pd.DataFrame(all_records)

    experiment_tag = (
        f"local_validation_queue_n200_alphaR10000"
        f"_n{NUM_SERVERS}"
        f"_{ARRIVAL_MODEL}"
        f"_C{str(ARRIVAL_SCALE_C).replace('.', 'p')}"
        f"_alpha{str(ARRIVAL_ALPHA).replace('.', 'p')}"
        f"_setup{SETUP_TIME}"
        f"_T{SIMULATION_TIME}"
        f"_{TURN_ON_MODE}"
        f"_{RESPONSE_METHOD}"
        f"_seed{len(GRID_SEARCH_SEEDS)}"
    )

    by_seed_path = RESULT_DIR / f"grid_search_by_seed_{experiment_tag}.csv"
    by_seed_df.to_csv(by_seed_path, index=False)

    summary_df = (
        by_seed_df
        .groupby(["evaluation_id", "T_i", "T_o"], as_index=False)
        .agg(
            mean_objective=("selected_objective", "mean"),
            std_objective=("selected_objective", "std"),
            num_seeds=("selected_objective", "count"),
            mean_power=("average_power", "mean"),
            mean_response_time=("average_response_time", "mean"),
            mean_system_size=("average_system_size", "mean"),
            mean_utilization=("utilization", "mean"),
            mean_ERP=("ERP", "mean")
        )
        .sort_values("mean_objective")
        .reset_index(drop=True)
    )

    summary_path = RESULT_DIR / f"grid_search_summary_{experiment_tag}.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nSaved grid search by-seed results to:")
    print(by_seed_path)

    print("\nSaved grid search summary to:")
    print(summary_path)

    return by_seed_df, summary_df


if __name__ == "__main__":

    print("Local grid validation")
    print("NUM_SERVERS:", NUM_SERVERS)
    print("ARRIVAL_MODEL:", ARRIVAL_MODEL)
    print("ARRIVAL_SCALE_C:", ARRIVAL_SCALE_C)
    print("ARRIVAL_ALPHA:", ARRIVAL_ALPHA)
    print("SIMULATION_TIME:", SIMULATION_TIME)
    print("SETUP_TIME:", SETUP_TIME)
    print("TURN_ON_MODE:", TURN_ON_MODE)
    print("RESPONSE_METHOD:", RESPONSE_METHOD)
    print("SEEDS:", GRID_SEARCH_SEEDS)

    if NUM_SERVERS != 200:
        print("\nWARNING: This validation is intended for NUM_SERVERS = 200.")

    all_records = run_grid_search(
        TURN_OFF_THRESHOLD_VALUES,
        TURN_ON_THRESHOLD_VALUES,
        GRID_SEARCH_SEEDS
    )

    by_seed_df, summary_df = save_grid_search_results(all_records)

    best_row = summary_df.iloc[0]

    print("\nGrid Search Best Result")
    print("Best T_i:", int(best_row["T_i"]))
    print("Best T_o:", int(best_row["T_o"]))
    print("Best mean objective:", float(best_row["mean_objective"]))

    print("\nTop 10 threshold pairs:")
    print(summary_df.head(10))
