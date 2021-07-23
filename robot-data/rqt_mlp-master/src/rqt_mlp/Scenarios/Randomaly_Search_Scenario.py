def Run_Scenario(scn_obj, number_scenarios):
    number_scenarios = int(number_scenarios)
    if number_scenarios > 0:
        run_one_scenario(scn_obj)

def run_one_scenario(scn_obj):
    import Search_Object as sobj
    sobj.Run_Scenario(scn_obj)