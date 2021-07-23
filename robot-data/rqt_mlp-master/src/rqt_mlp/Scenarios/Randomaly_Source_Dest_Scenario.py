import random
import time
import inspect, os
LOC_BOUND = 2
DISTANCE_LOWER_BOUND = 1
DISTANCE_UPPER_BOUND = 5
waiting_time = 10
# path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/Extentions/tmp/.scenarios_counter.tmp"

def Run_Scenario(scen_obj, number_scenarios):
    number_scenarios = int(number_scenarios)
    if number_scenarios > 0:
        run_once(scen_obj)

def run_once(scen_obj):
    source_x, source_y, angle, distance = generate_params()
    run_one_scenario(scen_obj, source_x, source_y, angle, distance)

def generate_params():
    import math
    source_x = random.uniform(-LOC_BOUND, LOC_BOUND)
    source_y = random.uniform(-LOC_BOUND, LOC_BOUND)
    angle = random.uniform(0, 2 * math.pi)
    distance = random.uniform(DISTANCE_LOWER_BOUND, DISTANCE_UPPER_BOUND)
    return source_x, source_y, angle, distance

def run_one_scenario(scen_obj, source_x, source_y, angle, distance):
    import Source_Dest_Scenario as sds
    sds.Run_Scenario(scen_obj, source_x, source_y, angle, distance)