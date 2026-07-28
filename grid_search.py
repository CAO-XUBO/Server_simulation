import os
import numpy as np
import pandas as pd
from scipy.stats import t

from simulator import server_simulator
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