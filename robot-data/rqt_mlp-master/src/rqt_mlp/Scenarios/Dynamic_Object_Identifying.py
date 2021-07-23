import os
import subprocess
import time
import rospy
from move_base_msgs.msg import MoveBaseActionResult

delaying_obj_appear = 5

def Run_Scenario(scen_obj, source_x, source_y, angle, distance):
    source_x, source_y, angle, distance = float(source_x), float(source_y), float(angle), float(distance)
    world = "empty.world"
    obj = "box"
    script_object = get_obj_script(obj, source_x, source_y, angle, distance)
    apply_scenario(scen_obj, source_x, source_y, angle, distance, world, script_object)

def get_obj_script(obj, source_x, source_y, angle, distance):
    import math, inspect, os
    model_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/Extentions/models/%s/%s.sdf" % (obj,obj) 
    # goal_x = source_x + distance * math.cos(angle)
    # goal_y = source_y + distance * math.sin(angle)
    # obj_x = (source_x + goal_x)
    # obj_y = (source_y + goal_y)
    dis = 1.2
    # if obj_x > 0:
    #   obj_x = obj_x + (dis * math.cos(angle))
    # else:
    #   obj_x = obj_x - (dis * math.sin(angle))
    # if obj_y > 0:
    #   obj_y = obj_y + (dis * math.sin(angle))
    # else:
    #   obj_y = obj_y - (dis * math.cos(angle))

    goal_x = source_x + (distance + dis) * math.cos(angle)
    goal_y = source_y + (distance + dis) * math.sin(angle)
    obj_x = (source_x + goal_x)
    obj_y = (source_y + goal_y)

    print "--------------------------------------------------------------------------------------------------------------------"
    print "angular = %s, obj_x = %s, obj_y = %s" % (angle, obj_x, obj_y)
    script_object =  "sleep %s && rosrun gazebo_ros spawn_model -file %s -sdf -model %s -y %s -x %s" % (delaying_obj_appear, model_path, obj, obj_y, obj_x)       
    return script_object

def apply_scenario(scen_obj, source_x, source_y, angle, distance, world, script_object):
    import Source_Dest_Scenario as sds
    sds.Run_Scenario(scen_obj, source_x, source_y, angle, distance, world, script_object)
