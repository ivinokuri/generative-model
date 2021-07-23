import random
LOC_BOUND = 0

def Run_Scenario(scen_obj, number_scenarios, goals, world="", mapping =""):
    number_scenarios = int(number_scenarios)
    if number_scenarios > 0:
        run_once(scen_obj, goals, world, mapping)

def run_once(scen_obj, goals, world, mapping):
    source_x, source_y, angle = generate_params()
    run_one_scenario(scen_obj, source_x, source_y, angle, goals, world, mapping)

def generate_params():
    source_x = random.uniform(-LOC_BOUND, LOC_BOUND)
    source_y = random.uniform(-LOC_BOUND, LOC_BOUND)
    angle = random.uniform(0, 0) #2 *	 math.pi)
    return source_x, source_y, angle

def run_one_scenario(scen_obj, source_x, source_y, angle, goals, world, mapping):
    import Apartment_Scenario as aparts
    aparts.Run_Scenario(scen_obj, source_x, source_y, angle, goals, world, mapping)