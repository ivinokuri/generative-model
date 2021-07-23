import os
import subprocess
import time
import rospy
from move_base_msgs.msg import MoveBaseActionResult
from move_base_msgs.msg import MoveBaseActionFeedback

start_scenarios_time = None
logging_msg = None
scenario_deadline = None

def Run_Scenario(scen_obj, source_x, source_y, angle, distance, world = "empty.world", script = ""):
    global start_scenarios_time, logging_msg, scenario_deadline
    source_x, source_y, angle, distance = float(source_x), float(source_y), float(angle), float(distance)
    # ros_launch = "roslaunch robotican_armadillo armadillo.launch lidar:=true move_base:=true " \
    #              "gmapping:=true gazebo:=true gui:=true world_name:=\"`rospack find rqt_mlp`/src/rqt_mlp/Scenarios/Extentions/worlds/%s\" " % world
    start_scenarios_time = rospy.Time.now().to_sec()
    scenario_deadline = calculate_scenario_deadline(distance, world)
    logging_msg = "%s\nsource x = %s, source y = %s, angle = %s, distance = %s, world = %s" % (script, round(source_x,2), round(source_y,2), round(angle,4), round(distance,2), world)

    ros_launch = "roslaunch robotican_armadillo armadillo.launch kinect2:=true lidar:=true move_base:=true " \
                 "gmapping:=true gazebo:=true world_name:=\"`rospack find rqt_mlp`/src/rqt_mlp/Scenarios/Extentions/worlds/%s\" " % world

    location = "x:=%s y:=%s Y:=%s" % (source_x, source_y, angle)
    launch_cmd = ros_launch + location
    subprocess.Popen(launch_cmd, shell=True)
    time.sleep(7)
    subprocess.Popen(script, shell=True)
    run_rviz()
    apply_statistics()
    apply_simulation(scen_obj, distance)
    scen_obj.generate_bag()

def run_rviz():
    rviz = "rosrun rviz rviz -d /home/lab/dwa.rviz"
    subprocess.Popen(rviz, shell=True)

def apply_statistics():
    enable_statistics = "rosparam set enable_statistics true"
    ros_profiler = "rosrun rosprofiler rosprofiler"
    subprocess.Popen(enable_statistics, shell=True)
    subprocess.Popen(ros_profiler, shell=True)

def apply_simulation(scen_obj, distance):
    goal = "rostopic pub /move_base_simple/goal geometry_msgs/PoseStamped \'{ header: { stamp: now, frame_id: \"/map\" }, " \
           "pose: { position: { x: %s, y: %s, z: 0}, orientation: { x: 0, y: 0, z: 0, w: 1 } } }\'" % (distance, 0)
    subprocess.Popen(goal, shell=True)
    rospy.Subscriber("/move_base/result", MoveBaseActionResult, is_arrived, scen_obj)
    # rospy.Subscriber("/move_base/feedback", MoveBaseActionFeedback, check_deadline, scen_obj)

def check_deadline(msg, scen_obj):
    now = rospy.Time.now().to_sec()
    current_duration = now - start_scenarios_time
    if current_duration > scenario_deadline:
        scen_obj.close_bag(delete=1)
        print "too much time"

def is_arrived(msg, scen_obj):
    global logging_msg, start_scenarios_time, scenario_deadline
    if msg.status.status == 3:
        end_time_scenarios = rospy.Time.now().to_sec()
        logging_params(logging_msg + ", duration = %s" % (round(end_time_scenarios - start_scenarios_time,2)))
        scen_obj.close_bag()
        print "shut down"

#in minutes
def calculate_scenario_deadline(distance, world):
    delta = 5
    if world == "empty.world":
        expected_duration = 9.1669 * distance + 11.614
    else:
        expected_duration = 1000
    return expected_duration + delta

def logging_params(logging_msg):
    import inspect
    import logging
    logging_file = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/Extentions/log/scenarios_logger.log"
    logger = logging.getLogger("scenarios_logger")

    hdlr = logging.FileHandler(logging_file)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.info(logging_msg)

    # if not os.path.isfile(logging_file):
    #     with open(logging_file, "w") as f:
    #         pass
    # with open(logging_file, "a") as f:
    #     str = "source x = %s, source y = %s, angle = %s, distance = %s, world = %s\n" % (source_x, source_y, angle, distance, world)
    #     f.write(str)



