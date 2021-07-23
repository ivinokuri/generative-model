#!/usr/bin/env python
## Phase 1

import inspect, os
from plugins.Analyzer import Analyzer

NO_COND = -9999
## for randomlay
scn_counter_path = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/Scenarios/Extentions/tmp/scenarios_counter.tmp"

topics_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/topics/"


class RunScenario:
    # ------------------------------------------------------------ constructor ------------------------------------------------------------

    # selected_scenario = {id = "", params = {x: value_x}}
    def __init__(self, bag_obj, export_bag, selected_scenario, selected_topics, action_id, time_interval, threshold, rule_filename, plp_filename, general_filename_read, specific_filename_read):
        must_topics = ['/host_diagnostic', '/node_diagnostic', '/statistics', "/move_base/feedback"]
        # must_topics = ['/host_statistics', '/node_statistics', '/statistics', "/move_base/feedback"]
        self.__restart_flag = False
        self.__bag_obj = bag_obj
        self.__export_bag = export_bag
        self.__selected_scenario = selected_scenario
        self.__selected_topics = list(set(selected_topics + must_topics))
        self.__action_id = action_id
        self.__time_interval = time_interval
        self.__threshold = threshold
        self.__rule_filename = rule_filename
        self.__plp_filename = plp_filename
        self.__general_filename_read = general_filename_read
        self.__specific_filename_read = specific_filename_read

    # ------------------------------------------------------------ getters ------------------------------------------------------------

    def get_export_bag(self):
        return self.__export_bag

    def get_selected_scenario(self):
        return self.__selected_scenario

    # ------------------------------------------------------------ setters ------------------------------------------------------------

    def add_temporal_filename(self, number):
        self.__export_bag = "%s_%s.bag" % (self.__export_bag[:-4], number)

    def turn_on_restart_flag(self):
        self.__restart_flag = True

    # ------------------------------------------------------------ functionality ------------------------------------------------------------

    # def apply_record_icon(self):
    #     self.__bag_obj.apply_record_icon()

    def run_record_scenario(self):
        scn_id = self.__selected_scenario['id']
        params = RunScenario.__get_scenarios_params(scn_id)
        operands = [self]
        for param in params:
            operands.append(self.__selected_scenario['params'][param])
        operator = (scenarios[scn_id])['function']
        operator(*operands)

    def get_topics_list(self):
        return self.__selected_topics

    def get_action_id(self):
        return self.__action_id

    def generate_bag(self):
        import subprocess
        # self.__bag_obj.record_bag(self.__export_bag, False, self.__selected_topics)
        # run = "gnome-terminal -e " + self.__plp_filename
        if self.__action_id != 1:
            goal1 = "rosbag record -O " + self.__export_bag
            for item in self.__selected_topics:
                goal1 += " " + item
            if self.__action_id == 3:
                goal1 += " --split --duration=" + str(self.__time_interval)
            print goal1
            subprocess.Popen(goal1, shell=True)
            if self.__action_id == 3:
                dir_path = os.path.dirname(os.path.realpath(self.__export_bag))
                with open(self.__general_filename_read) as f:
                    lines1 = f.read().splitlines()
                with open(self.__specific_filename_read) as f:
                    lines2 = f.read().splitlines()
                analyzer = Analyzer(dir_path, self.__time_interval, self.__threshold, self.__selected_topics, self.__rule_filename, lines1, lines2)
                import threading
                t = threading.Thread(target=analyzer.analyze(), args=())
                t.daemon = True
                t.start()
            print "##################### opened recorder ####################"
        else:
            return self.__action_id

    def close_bag(self):
        # path = ""
        # if delete != 0:
        #     path = self.__export_bag
        if not self.__restart_flag:
            if os.path.exists(scn_counter_path):
                os.remove(scn_counter_path)
        if self.__action_id != 1:
            self.__bag_obj.restart_recording(self.__export_bag)
        else:
            import subprocess
            restart_path = os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/Scenarios/Extentions/scripts/restart.sh"
            subprocess.Popen(restart_path, shell=True)
        # self.__bag_obj.restart_recording(self.__restart_flag, path)

    def get_topics(self):
        return list(self.__selected_topics)

    @staticmethod
    def __get_scenarios_params(scn_id):
        global scenarios
        print str(scenarios)
        return list(scenarios[scn_id]['params'])


# ------------------------------------------------------------ scenarios ------------------------------------------------------------

def source_destination_scenario(scn_obj, source_x, source_y, angle, distance):
    import Source_Dest_Scenario as sds
    sds.Run_Scenario(scn_obj, source_x, source_y, angle, distance)


def randomaly_source_destionation_scenario(scn_obj, number_simulations):
    import Randomaly_Source_Dest_Scenario as rsds
    create_restarting_file(scn_obj)
    # scn_obj.add_temporal_filename(number_simulations)
    rsds.Run_Scenario(scn_obj, number_simulations)


def obstacle_source_destination_scenario(scn_obj, source_x, source_y, angle, distance):
    import Obstacle_Source_Dest_Scenario as osds
    osds.Run_Scenario(scn_obj, source_x, source_y, angle, distance)


def randomaly_obstacle_source_destination_scenario(scn_obj, number_simulations):
    import Randomaly_Obstacle_Source_Dest_Scenario as rosds
    create_restarting_file(scn_obj)
    rosds.Run_Scenario(scn_obj, number_simulations)


def object_identifying_scenario(scn_obj, source_x, source_y, angle, distance):
    import Dynamic_Object_Identifying as doi
    doi.Run_Scenario(scn_obj, source_x, source_y, angle, distance)


def randomly_object_identifying_scenario(scn_obj, number_simulations):
    import Randomaly_Dynamic_Object_Identifying as rdoi
    create_restarting_file(scn_obj)
    rdoi.Run_Scenario(scn_obj, number_simulations)


def apartment_scenario(scn_obj, source_x, source_y, angle):
    import Apartment_Scenario as aparts
    aparts.Run_Scenario(scn_obj, source_x, source_y, angle)


def randomly_building_scenario(scn_obj, number_simulations):
    import Create_Goals
    import Randomaly_Apartment_Scenario as raparts
    create_restarting_file(scn_obj)
    goals = Create_Goals.building_goals()
    raparts.Run_Scenario(scn_obj, number_simulations, goals, world="buildings/building.world", mapping="building.yaml")


def randomly_obstacle_cans_scenario(scn_obj, number_simulations):
    import Create_Goals
    import Randomaly_Apartment_Scenario as raparts
    create_restarting_file(scn_obj)
    goals = Create_Goals.cans_goals()
    raparts.Run_Scenario(scn_obj, number_simulations, goals, world="cans/cans_1_obs.world", mapping="cans_1.yaml")


def randomly_cans_scenario(scn_obj, number_simulations):
    import Create_Goals
    import Randomaly_Apartment_Scenario as raparts
    create_restarting_file(scn_obj)
    goals = Create_Goals.cans_goals()
    raparts.Run_Scenario(scn_obj, number_simulations, goals, world="cans/cans_1.world", mapping="cans_1.yaml")


# def randomly_clean_room_scenario(scn_obj, number_simulations):
#     import Create_Goals
#     import Randomaly_Apartment_Scenario as raparts
#     create_restarting_file(scn_obj)
#     goals = Create_Goals.cans_goals()
#     raparts.Run_Scenario(scn_obj, number_simulations, goals, world="cans/clean_1.world", mapping="clean_1.yaml")

def randomly_corridor_scenario(scn_obj, number_simulations):
    import Create_Goals
    import Randomaly_Apartment_Scenario as raparts
    create_restarting_file(scn_obj)
    goals = Create_Goals.corridor_goals()
    raparts.Run_Scenario(scn_obj, number_simulations, goals, world="corridor/vert_corridor.world",
                         mapping="vert_corridor.yaml")


def randomly_stuff_corridor_scenario(scn_obj, number_simulations):
    import Create_Goals
    import Randomaly_Apartment_Scenario as raparts
    create_restarting_file(scn_obj)
    goals = Create_Goals.corridor_goals()
    raparts.Run_Scenario(scn_obj, number_simulations, goals, world="corridor/vert_corridor_stuff.world",
                         mapping="vert_corridor_stuff.yaml")


# def randomly_obs_corridor_scenario(scn_obj, number_simulations):
#     import Create_Goals
#     import Randomaly_Apartment_Scenario as raparts
#     create_restarting_file(scn_obj)
#     goals = Create_Goals.corridor_goals()
#     raparts.Run_Scenario(scn_obj, number_simulations, goals, world="corridor/vert_corridor.world", mapping="vert_corridor.yaml")

def randomly_obstacles_stuff_corridor_scenario(scn_obj, number_simulations):
    import Create_Goals
    import Randomaly_Apartment_Scenario as raparts
    create_restarting_file(scn_obj)
    goals = Create_Goals.corridor_goals()
    raparts.Run_Scenario(scn_obj, number_simulations, goals, world="corridor/vert_corridor_stuff_obs.world",
                         mapping="vert_corridor_stuff.yaml")


def search_can_scenario(scn_obj, number_simulations):
    import Randomaly_Search_Scenario as rss
    create_restarting_file(scn_obj)
    rss.Run_Scenario(scn_obj, number_simulations)

def arm_scenario(scn_obj, number_simulations):
    import Bag_Scenario as rss
    create_restarting_file(scn_obj)
    rss.Run_Scenario(scn_obj, number_simulations)


# ------------------------------------------------

def create_restarting_file(scn_obj):
    filename = scn_obj.get_export_bag()
    selected_scenario = scn_obj.get_selected_scenario()
    first_key = selected_scenario['params'].keys()[0]
    (selected_scenario['params'])[first_key] = int((selected_scenario['params'])[first_key])
    if (selected_scenario['params'])[first_key] < 1:
        # scn_obj.apply_record_icon()
        os.remove(scn_counter_path)
    else:
        import time
        if (selected_scenario['params'])[first_key] > 1:
            # print "hihihihihihi = %s " % (selected_scenario['params'])[first_key]

            # time.sleep(10)
            scn_obj.turn_on_restart_flag()
        time.sleep(10)
        (selected_scenario['params'])[first_key] = (selected_scenario['params'])[first_key] - 1
        # time.sleep(5)
        if os.path.exists(scn_counter_path):
            with open(scn_counter_path, 'r') as f:
                scn_number = int(f.read().splitlines()[1])
        else:
            scn_number = 0
        scn_obj.add_temporal_filename(scn_number)
        scn_number = scn_number + 1
        with open(scn_counter_path, 'w') as f:
            f.write(filename + '\n')
            f.write(str(scn_number) + '\n')
            f.write(str(selected_scenario))


# ------------------------------------------------------------ private functions ------------------------------------------------------------


# ----------------------------------------------------------------- global variables --------------------------------------------------------

scenarios = {}


# ----------------------------------------------------------------- global getters -----------------------------------------------------------



def get_scenarios_options():
    scenarios = create_scenarios()
    ret = {}
    for scn_id, scenario in scenarios.items():
        choise = dict(name=scenario["name"], params=list(scenario["params"]))
        ret[scn_id] = choise
    return ret


def get_topics_options():
    def get_topics_files(dir_path, suffix):
        files = [f for f in os.listdir(dir_path) if f[-len(suffix):] == suffix]
        return files

    files = get_topics_files(topics_directory, '.txt')
    ret = {}
    for f in files:
        topic_subject = f[:-4]  # get off '.txt' suffix
        content = ""
        with open(topics_directory + f) as read_f:
            content = [x.strip() for x in read_f.readlines()]  # remove whitespace
        ret[topic_subject] = content
    return ret


# ----------------------------------------------------------------- initializations ------------------------------------------------------

def create_scenarios():
    tmp_scenarios = {}
    # scenarios 1 -----------------------------------
    # scenario_id = 1
    # name = "walking without obstacles from specific source to specific destination"
    # ## param, label, condition
    # params = [('source x', '', NO_COND), ('source y', '', NO_COND), ('angle (rad)', '', NO_COND),
    #           ('distance', 'greater than zero', 0)]
    # function = source_destination_scenario
    # tmp_scenarios[scenario_id] = dict(name=name, params=params, function=function)
    # # scenarios 2 -----------------------------------
    # scenario_id = 2
    # name = "walking randomly without obstacles from source to destination"
    # params = [('number of simulations', 'greater than zero', 0)]
    # function = randomaly_source_destionation_scenario
    # tmp_scenarios[scenario_id] = dict(name=name, params=params, function=function)
    # # scenarios 3 -----------------------------------
    # scenario_id = 3
    # name = "walking with an obstacle from specific source to specific destination"
    # params = [('source x', '', NO_COND), ('source y', '', NO_COND), ('angle (rad)', '', NO_COND),
    #           ('distance', 'greater than 3', 3)]
    # function = obstacle_source_destination_scenario
    # tmp_scenarios[scenario_id] = dict(name=name, params=params, function=function)
    # # scenarios 4 -----------------------------------
    # scenario_id = 4
    # name = "walking randomly with an obstacle from source to destination"
    # params = [('number of simulations', 'greater than zero', 0)]
    # function = randomaly_obstacle_source_destination_scenario
    # tmp_scenarios[scenario_id] = dict(name=name, params=params, function=function)
    # # scenarios 5 -----------------------------------
    # scenario_id = 5
    # name = "object identifying"
    # params = [('source x', '', NO_COND), ('source y', '', NO_COND), ('angle (rad)', '', NO_COND),
    #           ('distance', 'greater than 3', 3)]
    # function = object_identifying_scenario
    # tmp_scenarios[scenario_id] = dict(name=name, params=params, function=function)
    # # scenarios 6 -----------------------------------
    # scenario_id = 6
    # name = "object identifying randomly"
    # params = [('number of simulations', 'greater than zero', 0)]
    # function = randomly_object_identifying_scenario
    # tmp_scenarios[scenario_id] = dict(name=name, params=params, function=function)
    # # scenarios 7 -----------------------------------
    # scenario_id = 7
    # name = "navigation into an apartment"
    # params = [('source x', '', NO_COND), ('source y', '', NO_COND), ('angle (rad)', '', NO_COND)]
    # function = apartment_scenario
    # tmp_scenarios[scenario_id] = dict(name=name, params=params, function=function)
    # # scenarios 8 -----------------------------------
    # scenario_id = 8
    # name = "navigation into a building randomaly"
    # params = [('number of simulations', 'greater than zero', 0)]
    # function = randomly_building_scenario
    # tmp_scenarios[scenario_id] = dict(name=name, params=params, function=function)
    # # scenarios 9 -----------------------------------
    # scenario_id = 9
    # name = "navigation into a corridor randomaly"
    # params = [('number of simulations', 'greater than zero', 0)]
    # function = randomly_corridor_scenario
    # tmp_scenarios[scenario_id] = dict(name=name, params=params, function=function)
    # # scenarios 10 -----------------------------------
    # scenario_id = 10
    # name = "navigation into a corridor with stuffs randomaly"
    # params = [('number of simulations', 'greater than zero', 0)]
    # function = randomly_stuff_corridor_scenario
    # tmp_scenarios[scenario_id] = dict(name=name, params=params, function=function)
    # # scenarios 11 -----------------------------------
    # scenario_id = 11
    # name = "navigation into a corridor with stuffs and obstacles randomaly"
    # params = [('number of simulations', 'greater than zero', 0)]
    # function = randomly_obstacles_stuff_corridor_scenario
    # tmp_scenarios[scenario_id] = dict(name=name, params=params, function=function)
    # # scenarios 12 -----------------------------------
    # scenario_id = 12
    # name = "navigation into cans randomaly"
    # params = [('number of simulations', 'greater than zero', 0)]
    # function = randomly_cans_scenario
    # tmp_scenarios[scenario_id] = dict(name=name, params=params, function=function)
    # # scenarios 13 -----------------------------------
    # scenario_id = 13
    # name = "navigation into cans with obstacles randomaly"
    # params = [('number of simulations', 'greater than zero', 0)]
    # function = randomly_obstacle_cans_scenario
    # tmp_scenarios[scenario_id] = dict(name=name, params=params, function=function)
    # scenarios 1 -----------------------------------
    scenario_id = 1
    name = "search object in the room"
    params = [('number of simulations', 'greater than zero', 0)]
    function = search_can_scenario
    tmp_scenarios[scenario_id] = dict(name=name, params=params, function=function)
    # scenarios 15 -----------------------------------
    # scenario_id = 15
    # name = "arm manipulation"
    # params = [('number of simulations', 'greater than zero', 0)]
    # function = arm_scenario
    # tmp_scenarios[scenario_id] = dict(name=name, params=params, function=function)
    return tmp_scenarios


def init_scenarios():
    global scenarios
    tmp_scenarios = create_scenarios()
    for scn_id, scenario in tmp_scenarios.items():
        params = (tmp_scenarios[scn_id])['params']
        new_params = []
        for param in params:
            new_params.append(param[0])
        (tmp_scenarios[scn_id])['params'] = new_params
    scenarios = tmp_scenarios


# to import dynamically a py_file from another directory
def import_dynamically(path):
    import sys
    sys.path.insert(0, path)


init_scenarios()
import_dynamically(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/Scenarios/")


# if __name__ == '__main__':
# selected_scenario = {"id" : 1, "params" : {'source x': 1 , 'source y': 2, 'destination x': 3, 'destination y': 4}}
# rs = RunScenario(selected_scenario, [])
# rs.generate_bag_file("asdasd")