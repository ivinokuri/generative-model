import os
import subprocess
import time
import rospy
from move_base_msgs.msg import MoveBaseActionResult

def Run_Scenario(scen_obj, source_x, source_y, angle, distance):
    source_x, source_y, angle, distance = float(source_x), float(source_y), float(angle), float(distance)
    world = "obstacle_map.world"
    obstacle = "box"
    obstacle_x, obstacle_y = calc_obstacle_loaction(source_x, source_y, angle, distance)
    apply_obstacle_world(world, obstacle, obstacle_x, obstacle_y)
    apply_scenario(scen_obj, source_x, source_y, angle, distance, world)

def calc_obstacle_loaction(source_x, source_y, angle, distance):
    import math
    goal_x = source_x + distance * math.cos(angle)
    goal_y = source_y + distance * math.sin(angle)
    obstacle_x = (source_x + goal_x) / 2
    obstacle_y = (source_y + goal_y) / 2
    return obstacle_x, obstacle_y

def apply_obstacle_world(world, obstacle, obstacle_x,obstacle_y):
    import inspect, os
    world_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/Extentions/worlds/%s" % world
    model_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/Extentions/models/%s/%s.sdf" % (obstacle,obstacle)
    obstacle_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/Extentions/models/%s" % obstacle
    pose = "%s %s -0.34 0 0 0" % (obstacle_x, obstacle_y)
    write_to_xml(world_path, './/include/uri', obstacle_path)
    write_to_xml(world_path, './/include/pose', pose)
    write_to_xml(model_path, './/script/uri', obstacle_path + "/material/scripts")
    write_to_xml(model_path, './/script/uri', obstacle_path + "/material/textures", 1)


def write_to_xml(path_file, xml_path, data, location = 0):
    from xml.etree import ElementTree as ET
    tree = ET.parse(path_file)
    (tree.findall(xml_path))[location].text = data
    tree.write(path_file)

def apply_scenario(scen_obj, source_x, source_y, angle, distance, world):
    import Source_Dest_Scenario as sds
    sds.Run_Scenario(scen_obj, source_x, source_y, angle, distance, world)