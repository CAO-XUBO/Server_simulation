'''
A policy contains three functions:
    initialize_server_state: defines the initial state of the server
    should_start_setup: defines whether the policy should start setup(turn on)
    should_turn_off: defines whether the policy should turn off
'''

import numpy as np

def count_state(server_state, target_state):
    return sum(1 for state in server_state if state == target_state)

def choose_off_server(server_state):
    for i, state in enumerate(server_state):
        if state == "OFF":
            return i
    return None

## NEVEROFF Policy
def should_start_setup_neveroff(queue_length, server_state, turn_on_threshold = None):
    '''
    The turn-on rule for NEVEROFF policy
    '''
    return False

def should_turn_off_neveroff(server_id, server_state, turn_off_threshold = None):
    '''
    The turn-off rule for NEVEROFF policy
    '''
    return False

def initialize_server_state_neveroff(Num_server, turn_off_threshold=None):
    return ["IDLE"] * Num_server

## INSTANTOFF Policy
def should_start_setup_instantoff(queue_length, server_state, turn_on_threshold = None):
    '''
    The turn-on rule for INSTANTOFF policy
    '''

    Num_idle_servers = count_state(server_state, "IDLE")
    Num_setup_servers = count_state(server_state, "SETUP")
    Num_off_servers = count_state(server_state, "OFF")

    if queue_length == 0:
        return False

    if Num_idle_servers > 0:
        return False

    if Num_off_servers == 0:
        return False

    if Num_setup_servers >= queue_length:
        return False

    return True

def should_turn_off_instantoff(server_id, server_state, turn_off_threshold = None):
    '''
    The turn-off rule for INSTANTOFF policy
    '''
    return server_state[server_id] == "IDLE"

def initialize_server_state_instantoff(Num_server, turn_off_threshold=None):
    return ["OFF"] * Num_server

## THRESHOLD Policy
def should_start_setup_threshold(queue_length, server_state, turn_on_threshold):
    """
    The turn-on rule for Threshold policy.

    T_o >= 0:
        idle-based turn-on rule.

    T_o < 0:
        queue-based turn-on rule.
    """

    Num_idle_servers = count_state(server_state, "IDLE")
    Num_off_servers = count_state(server_state, "OFF")

    if Num_off_servers == 0:
        return False

    # Case I: T_o >= 0, use number of idle servers
    if turn_on_threshold >= 0:
        return Num_idle_servers <= turn_on_threshold

    # Case II: T_o < 0, use queue length
    queue_threshold = abs(turn_on_threshold)
    return queue_length >= queue_threshold

def should_turn_off_threshold(server_id, server_state, turn_off_threshold):
    '''
    The turn-off rule for Threshold policy
    turn_off_threshold: T_i
    '''
    # Cannot turn off a busy server
    if server_state[server_id] != "IDLE":
        return False

    Num_idle_servers = count_state(server_state, "IDLE")

    if Num_idle_servers > turn_off_threshold:
        return True
    else:
        return False

def initialize_server_state_threshold(Num_server, turn_off_threshold):
    initial_idle_servers = min(turn_off_threshold, Num_server)
    return ["IDLE"] * initial_idle_servers + ["OFF"] * (Num_server - initial_idle_servers)

def get_policy_functions(policy):
    if policy == "NEVEROFF":
        return {
            "initialize_server_state": initialize_server_state_neveroff,
            "should_start_setup": should_start_setup_neveroff,
            "should_turn_off": should_turn_off_neveroff,
            "choose_off_server": choose_off_server
        }

    elif policy == "INSTANTOFF":
        return {
            "initialize_server_state": initialize_server_state_instantoff,
            "should_start_setup": should_start_setup_instantoff,
            "should_turn_off": should_turn_off_instantoff,
            "choose_off_server": choose_off_server
        }

    elif policy == "THRESHOLD":
        return {
            "initialize_server_state": initialize_server_state_threshold,
            "should_start_setup": should_start_setup_threshold,
            "should_turn_off": should_turn_off_threshold,
            "choose_off_server": choose_off_server
        }

    else:
        raise ValueError("Unknown policy")