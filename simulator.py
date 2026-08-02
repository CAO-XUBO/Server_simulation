import numpy as np
from src.Config import *
from src.policies import get_policy_functions
from src.arrival_process import generate_next_arrival_time, get_arrival_rate

def count_busy_servers(server_state):
    busy_servers = sum(1 for state in server_state if state == "BUSY")
    return busy_servers

def count_server_states(server_state):
    return {
        "BUSY": sum(1 for state in server_state if state == "BUSY"),
        "IDLE": sum(1 for state in server_state if state == "IDLE"),
        "SETUP": sum(1 for state in server_state if state == "SETUP"),
        "OFF": sum(1 for state in server_state if state == "OFF"),
    }

# def calculate_current_power(server_state):
#     current_power = 0
#     for state in server_state:
#         if state == "BUSY":
#             current_power += P_BUSY
#         elif state == "IDLE":
#             current_power += P_IDLE
#         elif state == "OFF":
#             current_power += P_OFF
#         elif state == "SETUP":
#             current_power += P_SETUP
#         else:
#             raise ValueError("Unknown state")
#     return current_power

# def get_arrival_rate(Num_server, base_arrival_rate, arrival_model = "fixed", C = 0.3, alpha = 0.5):
#     '''
#     arrival_model
#     fixed: a fixed arrival rate lambda
#     fixed_scaling: lambda^n = n - C * n^alpha
#     '''
#     if arrival_model == "fixed":
#         return base_arrival_rate
#
#     elif arrival_model == "fixed_scaling":
#         arrival_rate = Num_server - C * (Num_server ** alpha)
#
#         if arrival_rate <= 0:
#             raise ValueError("Arrival rate must be positive")
#         return arrival_rate
#
#     else:
#         raise ValueError("Unknown arrival mode")

def start_service(server_id, arrival_time, current_time, service_rate, server_state, current_customer_arrival,response_method, event_calendar):

    # Set the server to busy state
    server_state[server_id] = "BUSY"

    if response_method == "exact":
        current_customer_arrival[server_id] = arrival_time

    departure_time = current_time + np.random.exponential(1 / service_rate)
    event_calendar.append((departure_time, "departure", server_id))

def start_setup(server_id, current_time, setup_time, server_state, event_calendar):

    # Set the server to setup state
    server_state[server_id] = "SETUP"

    setup_complete_time = current_time + setup_time
    event_calendar.append((setup_complete_time, "setup_complete", server_id))

def apply_setup_policy(central_queue, current_time, setup_time,
                       server_state, event_calendar, policy_functions,
                       turn_on_threshold):
    if policy_functions["should_start_setup"](central_queue, server_state, turn_on_threshold):
        off_server = policy_functions["choose_off_server"](server_state)

        if off_server is not None:
            start_setup(
                off_server,
                current_time,
                setup_time,
                server_state,
                event_calendar
            )

def apply_turn_off_policy(server_id, server_state, policy_functions,
                          turn_off_threshold):
    if policy_functions["should_turn_off"](server_id, server_state, turn_off_threshold):
        server_state[server_id] = "OFF"

def find_idle_server(server_state):
    for i, state in enumerate(server_state):
        if state == "IDLE":
            return i
    return None

def dispatch_jobs_to_idle_servers(central_queue, current_time, service_rate, response_method,
                                  server_state, current_customer_arrival,
                                  event_calendar):
    added_waiting_time = 0
    added_started_service = 0

    while len(central_queue) > 0:
        idle_server = find_idle_server(server_state)

        if idle_server is None:
            break

        if response_method == "exact":
            arrival_time = central_queue.pop(0)
        else:
            arrival_time = None
            central_queue -= 1

        waiting_time = current_time - arrival_time
        added_waiting_time += waiting_time
        added_started_service += 1

        start_service(idle_server, arrival_time, current_time, service_rate, server_state, current_customer_arrival,
                      event_calendar)

    return added_waiting_time, added_started_service

def get_queue_length(central_queue, response_method):
    if response_method == "exact":
        return len(central_queue)
    elif response_method == "little":
        return central_queue
    else:
        raise ValueError("response_method must be either 'exact' or 'little'.")

def server_simulator(Num_server = 5,
                     arrival_rate = 1,
                     service_rate = 1.5,
                     timesteps = 100,
                     setup_time = SETUP_TIME,
                     policy = "NEVEROFF",
                     turn_off_threshold = 5,
                     turn_on_threshold = -3,
                     arrival_model = ARRIVAL_MODEL,
                     arrival_scale_C = ARRIVAL_SCALE_C,
                     arrival_alpha = ARRIVAL_ALPHA,
                     arrival_amplitude = ARRIVAL_AMPLITUDE,
                     response_method = "little",
                     seed = 42):
    '''
    Num_server: The number of server in the system
    arrival_rate: The arrival rate lambda
    service_rate: The service rate mu
    timesteps: Simulation times
    setup_time: The setup time
    policy: The policy function
    turn_off_threshold: The turn off threshold T_i
    turn_on_threshold: The turn on threshold T_o
    arrival_model: The arrival model (fixed, fixed_scaling, time_varying_scaling)
    return: Average_System_Size L, Utilization rho, Average_Power, Average_Waiting_Time, Average_Response_Time_Exact,
     Average_Response_Time_Little, ERP_Exact, ERP_Little
    '''

    # Set the random seed
    if seed is not None:
        np.random.seed(seed)

    ## Initialisation
    policy_functions = get_policy_functions(policy)
    server_state = policy_functions["initialize_server_state"](
        Num_server,
        turn_off_threshold
    )
    # Central queue
    if response_method == "exact":
        central_queue = []
        current_customer_arrival = [None] * Num_server
        total_waiting_time = 0.0
        Num_started_service = 0
        total_response_time = 0.0
    elif response_method == "little":
        central_queue = 0
        current_customer_arrival = None
        total_waiting_time = np.nan
        Num_started_service = 0
        total_response_time = np.nan
    else:
        raise ValueError("response_method must be either 'exact' or 'little'.")

    Area_server_state = 0 # AB
    Area_users = 0  # AQ

    Num_completed_users = 0
    current_customer_arrival = [None] * Num_server

    # Initialise the time on each state
    busy_server_time = 0.0
    idle_server_time = 0.0
    setup_server_time = 0.0
    off_server_time = 0.0

    ## Initialise the event calendar
    # Schedule the first arrival event
    current_time = 0

    first_arrival_time = generate_next_arrival_time(
        current_time=0,
        Num_server=Num_server,
        base_arrival_rate=arrival_rate,
        arrival_model=arrival_model,
        C=arrival_scale_C,
        alpha=arrival_alpha,
        timesteps=timesteps,
        arrival_amplitude=arrival_amplitude
    )
    event_calendar = [(timesteps, "termination", None)]

    if first_arrival_time is not None:
        event_calendar.append((first_arrival_time, "arrival", None))

    while True:
        # Find the next event and delete it from the event calendar
        next_index = min(range(len(event_calendar)), key=lambda i: event_calendar[i][0])
        event_time, event_type, server_id = event_calendar.pop(next_index)

        delta_time = event_time - current_time

        state_counts = count_server_states(server_state)

        busy_server = state_counts["BUSY"]
        idle_server = state_counts["IDLE"]
        setup_server = state_counts["SETUP"]
        off_server = state_counts["OFF"]

        queue_length = get_queue_length(central_queue, response_method)
        system_size = busy_server + queue_length
        Area_users += delta_time * system_size

        # Update average number of jobs and utilization area
        Area_users += delta_time * system_size
        Area_server_state += delta_time * busy_server

        # Update server-state time
        busy_server_time += delta_time * busy_server
        idle_server_time += delta_time * idle_server
        setup_server_time += delta_time * setup_server
        off_server_time += delta_time * off_server


        # Update the current time
        current_time = event_time

        if event_type == "arrival":
            # arrival event
            arrival_time = current_time

            # New jobs enter the central queue
            if response_method == "exact":
                central_queue.append(current_time)
            else:
                central_queue += 1
            # Dispatcher jobs to idle server
            added_waiting_time, added_started_service = dispatch_jobs_to_idle_servers(
                central_queue,
                current_time,
                service_rate,
                server_state,
                current_customer_arrival,
                event_calendar)

            total_waiting_time += added_waiting_time
            Num_started_service += added_started_service

            apply_setup_policy(central_queue,
                               current_time,
                               setup_time,
                               server_state,
                               event_calendar,
                               policy_functions,
                               turn_on_threshold)

            # Schedule the next arrival time
            next_arrival_time = generate_next_arrival_time(
                current_time=current_time,
                Num_server=Num_server,
                base_arrival_rate=arrival_rate,
                arrival_model=arrival_model,
                C=arrival_scale_C,
                alpha=arrival_alpha,
                timesteps=timesteps,
                arrival_amplitude=arrival_amplitude
            )

            if next_arrival_time is not None:
                event_calendar.append((next_arrival_time, "arrival", None))

        elif event_type == "departure":

            Num_completed_users += 1

            if response_method == "exact":
                response_time = current_time - current_customer_arrival[server_id]
                total_response_time += response_time
                current_customer_arrival[server_id] = None

            # Server becomes idle after setup is completed
            server_state[server_id] = "IDLE"

            # Dispatch another job to the idle server
            added_waiting_time, added_started_service = dispatch_jobs_to_idle_servers(
                central_queue,
                current_time,
                service_rate,
                server_state,
                current_customer_arrival,
                event_calendar
            )

            total_waiting_time += added_waiting_time
            Num_started_service += added_started_service

            # If the server is still idle after dispatching, apply policy
            if server_state[server_id] == "IDLE":
                apply_turn_off_policy(server_id, server_state, policy_functions, turn_off_threshold)

            # Check the turn-on threshold T_o
            apply_setup_policy(
                central_queue,
                current_time,
                setup_time,
                server_state,
                event_calendar,
                policy_functions,
                turn_on_threshold
            )

        elif event_type == "setup_complete":
            # Server becomes idle after completing a job
            server_state[server_id] = "IDLE"

            # Dispatch another job to the idle server
            added_waiting_time, added_started_service = dispatch_jobs_to_idle_servers(
                central_queue,
                current_time,
                service_rate,
                server_state,
                current_customer_arrival,
                event_calendar
            )

            total_waiting_time += added_waiting_time
            Num_started_service += added_started_service

            # If the server is still idle after dispatching, apply policy
            if server_state[server_id] == "IDLE":
                apply_turn_off_policy(server_id, server_state, policy_functions, turn_off_threshold)

            # Check the turn-on threshold T_o
            apply_setup_policy(
                central_queue,
                current_time,
                setup_time,
                server_state,
                event_calendar,
                policy_functions,
                turn_on_threshold
            )

        elif event_type == "termination":
            Average_System_Size = Area_users/timesteps # L
            Utilization = Area_server_state / (Num_server * timesteps) # rho

            # Calculate the expected energy consumption with decomposition
            busy_energy = P_BUSY * busy_server_time
            idle_energy = P_IDLE * idle_server_time
            setup_energy = P_SETUP * setup_server_time
            off_energy = P_OFF * off_server_time

            total_energy = (
                    busy_energy
                    + idle_energy
                    + setup_energy
                    + off_energy
            )
            Average_Power = total_energy / timesteps

            if Num_started_service > 0:
                Average_Waiting_Time = total_waiting_time / Num_started_service
            else:
                Average_Waiting_Time = 0
            if Num_completed_users > 0:
                Average_Response_Time_Exact = total_response_time / Num_completed_users
            else:
                Average_Response_Time_Exact = 0

            # Calculate the realised throughput
            throughput = Num_completed_users / timesteps

            # Estimate average response time using Little's Law
            if throughput > 0:
                Average_Response_Time_Little = (
                        Average_System_Size / throughput
                )
            else:
                Average_Response_Time_Little = np.nan

            # Cost functions
            # E[R]: Expected Response Time
            # E[E]: Expected Energy Consumption
            # ERP := E[R]*E[E]
            ERP_Exact = Average_Power * Average_Response_Time_Exact
            ERP_Little = Average_Power * Average_Response_Time_Little

            # Linear_cost_function = Weight * E[R] + E[E]
            Objective_Exact = RESPONSE_TIME_WEIGHT * Average_Response_Time_Exact + Average_Power
            Objective_Little = RESPONSE_TIME_WEIGHT * Average_Response_Time_Little + Average_Power

            return (
                Average_System_Size,
                Utilization,
                Average_Power,
                Average_Response_Time,
                ERP,
                Objective,
                Num_completed_users
            )

if __name__ == "__main__":

    policy = "NEVEROFF"
    (Average_System_Size, Utilization, Average_Power, Average_Waiting_Time, Average_Response_Time_Exact,
     Average_Response_Time_Little, ERP_Exact, ERP_Little, Objective_Exact, Objective_Little)  = server_simulator(
        Num_server=NUM_SERVERS,
        arrival_rate=ARRIVAL_RATE,
        service_rate=SERVICE_RATE,
        timesteps=SIMULATION_TIME,
        setup_time=SETUP_TIME,
        policy=policy,  # "INSTANTOFF", "NEVEROFF", "THRESHOLD"
        turn_off_threshold=5,
        turn_on_threshold=-3,
        arrival_model="fixed_scaling",
        arrival_scale_C=0.3,
        arrival_alpha=0.5,
        seed=42
    )

    print("Simulation Finished")
    print("Policy:", policy)
    print("The Average System Size:", Average_System_Size)
    print("Utilization:", Utilization)
    print("Average Power:", Average_Power)
    print("Average Waiting Time:", Average_Waiting_Time)
    print("Average Response Time (Exact):", Average_Response_Time_Exact)
    print("Average Response Time (Little's law):", Average_Response_Time_Little)
    print("ERP (Exact):", ERP_Exact)
    print("ERP (Little's law):", ERP_Little)
    print("Objective (Exact):", Objective_Exact)
    print("Objective (Little's law):", Objective_Little)
