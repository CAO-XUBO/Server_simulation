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

def start_service(server_id,
                  arrival_time,
                  current_time,
                  service_rate,
                  server_state,
                  current_customer_arrival,
                  response_method,
                  event_calendar):

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

def apply_setup_policy(queue_length, current_time, setup_time,
                       server_state, event_calendar, policy_functions,
                       turn_on_threshold):
    if policy_functions["should_start_setup"](queue_length, server_state, turn_on_threshold):
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

def dispatch_jobs_to_idle_servers(central_queue,
                                  current_time,
                                  service_rate,
                                  response_method,
                                  server_state,
                                  current_customer_arrival,
                                  event_calendar):

    while get_queue_length(central_queue, response_method) > 0:
        idle_server = find_idle_server(server_state)

        if idle_server is None:
            break

        if response_method == "exact":
            arrival_time = central_queue.pop(0)
        else:
            arrival_time = None
            central_queue -= 1

        start_service(
            idle_server,
            arrival_time,
            current_time,
            service_rate,
            server_state,
            current_customer_arrival,
            response_method,
            event_calendar
        )

    return central_queue

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
    Num_server: The number of servers in the system
    arrival_rate: The arrival rate lambda
    service_rate: The service rate mu
    timesteps: Simulation time horizon
    setup_time: The setup time
    policy: The control policy
    turn_off_threshold: The turn-off threshold T_i
    turn_on_threshold: The turn-on threshold T_o
    arrival_model: The arrival model
    response_method:
        "exact"  -> compute response time from completed jobs
        "little" -> compute response time using Little's law

    return:
        Average_System_Size,
        Utilization,
        Average_Power,
        Average_Response_Time,
        ERP,
        Objective,
        Num_completed_users
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
        total_response_time = 0.0

    elif response_method == "little":
        central_queue = 0
        current_customer_arrival = None

    else:
        raise ValueError("response_method must be either 'exact' or 'little'.")

    Area_server_state = 0 # AB
    Area_users = 0  # AQ

    Num_completed_users = 0

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
            # New jobs enter the central queue
            if response_method == "exact":
                central_queue.append(current_time)
            else:
                central_queue += 1
            # Dispatcher jobs to idle server
            central_queue = dispatch_jobs_to_idle_servers(
                central_queue,
                current_time,
                service_rate,
                response_method,
                server_state,
                current_customer_arrival,
                event_calendar
            )

            queue_length = get_queue_length(central_queue, response_method)

            apply_setup_policy(
                queue_length,
                current_time,
                setup_time,
                server_state,
                event_calendar,
                policy_functions,
                turn_on_threshold
            )

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

            # Server becomes idle after completing a job
            server_state[server_id] = "IDLE"

            # Dispatch another job to the idle server
            central_queue = dispatch_jobs_to_idle_servers(
                central_queue,
                current_time,
                service_rate,
                response_method,
                server_state,
                current_customer_arrival,
                event_calendar
            )

            # If the server is still idle after dispatching, apply policy
            if server_state[server_id] == "IDLE":
                apply_turn_off_policy(server_id, server_state, policy_functions, turn_off_threshold)

            # Check the turn-on threshold T_o

            queue_length = get_queue_length(central_queue, response_method)

            apply_setup_policy(
                queue_length,
                current_time,
                setup_time,
                server_state,
                event_calendar,
                policy_functions,
                turn_on_threshold
            )

        elif event_type == "setup_complete":
            # Server becomes idle after setup is completed
            server_state[server_id] = "IDLE"

            # Dispatch another job to the idle server
            central_queue = dispatch_jobs_to_idle_servers(
                central_queue,
                current_time,
                service_rate,
                response_method,
                server_state,
                current_customer_arrival,
                event_calendar
            )


            # If the server is still idle after dispatching, apply policy
            if server_state[server_id] == "IDLE":
                apply_turn_off_policy(server_id, server_state, policy_functions, turn_off_threshold)

            # Check the turn-on threshold T_o

            queue_length = get_queue_length(central_queue, response_method)

            apply_setup_policy(
                queue_length,
                current_time,
                setup_time,
                server_state,
                event_calendar,
                policy_functions,
                turn_on_threshold
            )

        elif event_type == "termination":
            Average_System_Size = Area_users / timesteps
            Utilization = Area_server_state / (Num_server * timesteps)

            busy_energy = P_BUSY * busy_server_time
            idle_energy = P_IDLE * idle_server_time
            setup_energy = P_SETUP * setup_server_time
            off_energy = P_OFF * off_server_time

            total_energy = busy_energy + idle_energy + setup_energy + off_energy
            Average_Power = total_energy / timesteps

            actual_arrival_rate = get_arrival_rate(
                Num_server=Num_server,
                base_arrival_rate=arrival_rate,
                arrival_model=arrival_model,
                C=arrival_scale_C,
                alpha=arrival_alpha,
                current_time=0,
                timesteps=timesteps,
                arrival_amplitude=arrival_amplitude
            )

            if response_method == "exact":
                if Num_completed_users > 0:
                    Average_Response_Time = total_response_time / Num_completed_users
                else:
                    Average_Response_Time = np.nan

            elif response_method == "little":
                if actual_arrival_rate > 0:
                    Average_Response_Time = Average_System_Size / actual_arrival_rate
                else:
                    Average_Response_Time = np.nan

            ERP = Average_Power * Average_Response_Time
            Objective = RESPONSE_TIME_WEIGHT * Average_Response_Time + Average_Power

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
        turn_off_threshold=5,
        turn_on_threshold=-3,
        arrival_model="fixed_scaling",
        arrival_scale_C=0.3,
        arrival_alpha=0.5,
        response_method="little",
        seed=42
    )

    print("Simulation Finished")
    print("Policy:", policy)
    print("Response method:", "little")
    print("Average System Size:", Average_System_Size)
    print("Utilization:", Utilization)
    print("Average Power:", Average_Power)
    print("Average Response Time:", Average_Response_Time)
    print("ERP:", ERP)
    print("Objective:", Objective)
    print("Number of completed jobs:", Num_completed_users)