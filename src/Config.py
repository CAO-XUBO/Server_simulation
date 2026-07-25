# Hyperparameters

# Energy consumption
P_BUSY = 240
P_IDLE = 150
P_OFF = 0
P_SETUP = 150

## Experimental Settings
#Arrival Process
ARRIVAL_MODEL = "fixed_scaling"
ARRIVAL_RATE = 1.5 # lambda
ARRIVAL_SCALE_C = 0.3 # C
ARRIVAL_ALPHA = 0.5 # alpha
ARRIVAL_AMPLITUDE = 0.5 #

# SERVICE DISTRIBUTION
SERVICE_RATE = 1.0 # mu
SIMULATION_TIME = 10000
NUM_SERVERS = 100
SEED = 42
SETUP_TIME = 200

# Linear cost function indicator
RESPONSE_TIME_WEIGHT = 1

LARGE_PENALTY = 1e12