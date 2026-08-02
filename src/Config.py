# Hyperparameters

# Energy consumption
P_BUSY = 240
P_IDLE = 150
P_OFF = 0
P_SETUP = 150

## Experimental Settings
#Arrival Process

# arrival_model
# fixed: a fixed arrival rate lambda
# fixed_scaling: lambda^n = n - C * n^alpha
# time_varying_scaling: lambda^n(t) = n - C * n^alpha * (1 + A * sin(2*pi*t/T))

ARRIVAL_MODEL = "fixed_scaling"
ARRIVAL_RATE = 1.5 # lambda
ARRIVAL_SCALE_C = 2 # C
ARRIVAL_ALPHA = 0.5 # alpha
ARRIVAL_AMPLITUDE = 0.5 #

# SERVICE DISTRIBUTION
SERVICE_RATE = 1.0 # mu
SIMULATION_TIME = 100
NUM_SERVERS = 10
SEED = 42
SETUP_TIME = 5

# Linear cost function indicator
RESPONSE_TIME_WEIGHT = 1

LARGE_PENALTY = 1e12