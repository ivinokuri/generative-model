import os
import subprocess
import time
import rospy
from move_base_msgs.msg import MoveBaseActionResult
from move_base_msgs.msg import MoveBaseActionGoal
from geometry_msgs.msg import PoseStamped
import random


SLEEPING_TIME = random.uniform(15, 22) #15
rosbag_process = None
start_scenarios_time = None
logging_msg = None

def Run_Scenario(scen_obj, source_x, source_y, angle, goals, world = "", mapping = ""):
    global start_scenarios_time, logging_msg
    # print world
    time.sleep(10)
    source_x, source_y, angle = float(source_x), float(source_y), float(angle)
    start_scenarios_time = rospy.Time.now().to_sec()
    logging_msg = "source x = %s, source y = %s, angle = %s, mapping = %s, world = %s" % (round(source_x,2), round(source_y,2), round(angle,4), mapping, world)
    ros_launch = "roslaunch robotican_armadillo armadillo.launch kinect2:=true lidar:=true move_base:=true " \
		 "moveit:=false use_sim_time:=true robot_localization:=true arm:=false controllers:=true " \
                 "amcl:=true hector_slam:=false gmapping:=false gazebo:=true have_map_file:=true map_file:=\"`rospack find rqt_mlp`/src/rqt_mlp/Scenarios/Extentions/maps/%s\"  world_name:=\"`rospack find rqt_mlp`/src/rqt_mlp/Scenarios/Extentions/worlds/%s\" " % (mapping ,world)

    location = "x:=%s y:=%s Y:=%s" % (source_x, source_y, angle)
    launch_cmd = ros_launch + location
    subprocess.Popen(launch_cmd, shell=True)
    apply_statistics()
    # raw_input("Press Enter to continue...")
    time.sleep(random.uniform(15, 22))
    run_rviz()
    #run_attacker()
    apply_diagnostic()
    apply_simulation(scen_obj, goals)
    #record_start(scen_obj)
    scen_obj.generate_bag()

#def record_start(scen_obj):
  #topics = scen_obj.get_topics()
  #rosbag_cmd = "rosbag record " + reduce(lambda acc,curr: acc + " " + curr, topics, " ")
  #return subprocess.Popen(rosbag_cmd, shell=True)

#def record_end(scen_obj):
  #import rosnode, rosgraph
  #nodes = rosnode.get_nodes_by_machine(rosgraph.network.get_host_name())
  #record_node = filter(lambda node_name: "record" in node_name, nodes)[0]
  #rosbag_cmd = "rosnode kill " + record_node
  #subprocess.Popen(rosbag_cmd, shell=True)
  
def run_attacker():
    rviz = "roslaunch robotican_demos attacker.launch" #-d /home/lab/dwa.rviz"
    subprocess.Popen(rviz, shell=True)

def run_rviz():
    rviz = "rosrun rviz rviz" #-d /home/lab/dwa.rviz"
    subprocess.Popen(rviz, shell=True)

def apply_diagnostic():
    ros_profiler = "rosrun rosdiagnostic rosdiagnostic"
    subprocess.Popen(ros_profiler, shell=True)
    pass
  
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


def apply_statistics():
    enable_statistics = "rosparam set enable_statistics true"
    subprocess.Popen(enable_statistics, shell=True)
    #ros_profiler = "rosrun rosprofiler rosprofiler"    
    #subprocess.Popen(ros_profiler, shell=True)
 
next_goal = 1
waiting = None
def when_arrived(msg, args):
  global next_goal, logging_msg, start_scenarios_time, waiting
  scen_obj, goals = args[0], args[1]
  if msg.status.status == 3 and next_goal >= len(goals): #end
    end_time_scenarios = rospy.Time.now().to_sec()
    logging_params(logging_msg + ", duration = %s" % (round(end_time_scenarios - start_scenarios_time,2)))
    waiting.unregister()
    #pub.unregister()
    # terminate()
    # time.sleep(2)
    scen_obj.close_bag()
    #time.sleep(10)
    print "shut down"
  elif msg.status.status == 3:
    publish(create_goal_msg(goals[next_goal], next_goal))
    #pub.publish(create_goal_msg(goals[next_goal], next_goal))
    next_goal = next_goal + 1

def terminate():
  clean_script = "/home/matansar/catkin_ws/src/rqt_mlp/src/rqt_mlp/Scenarios/Extentions/scripts/restart.sh"
  subprocess.Popen(clean_script, shell=True)
  
#def create_simple_goal_msg(goal, seq):
  #x = goal[0]
  #y = goal[1]
  #msg = PoseStamped()
  #msg.header.seq = seq
  #msg.header.stamp = rospy.Time.now()
  #msg.header.frame_id = 'map'
  #msg.pose.position.x = x
  #msg.pose.position.y = y
  #return msg
  
def create_goal_msg(goal, seq):
  import math, random, tf
  x = goal[0]
  y = goal[1]
  orient = tf.transformations.quaternion_from_euler(0, 0, random.uniform(0, 2 * math.pi))
  goal = "\'{ header: { stamp: now, frame_id: \"/map\" }, " \
	   "goal: { target_pose: { header:{ seq: %s, stamp: now, frame_id: \"/map\" }, pose: " \
	   "{ position: {x: %s, y: %s}, orientation: { x: %s, y: %s, z: %s, w: %s} } } } }\'" % (seq, x, y, orient[0], orient[1], orient[2], orient[3])
  return goal
  #msg = MoveBaseActionGoal()
  #msg.header.seq = seq
  #t = rospy.Time() #.now()
  #msg.header.stamp = t
  #msg.header.frame_id = ''
  #msg.goal.target_pose.header.seq = seq
  #msg.goal.target_pose.header.stamp = t
  #msg.goal.target_pose.header.frame_id = 'map'
  #msg.goal.target_pose.pose.position.x = x
  #msg.goal.target_pose.pose.position.y = y
  #orient = tf.transformations.quaternion_from_euler(0, 0, random.uniform(0, 2 * math.pi))
  #msg.goal.target_pose.pose.orientation.x = orient[0]
  #msg.goal.target_pose.pose.orientation.y = orient[1]
  #msg.goal.target_pose.pose.orientation.z = orient[2]
  #msg.goal.target_pose.pose.orientation.w = orient[3]
  ##print msg.goal.target_pose.pose.orientation.x
  ##print msg.goal.target_pose.pose.orientation.y
  ##print msg.goal.target_pose.pose.orientation.z
  ##print msg.goal.target_pose.pose.orientation.w
  ##time.sleep(20)
  ##msg.goal.target_pose.pose.orientation.z = 1
  #return msg
  
def publising_goals(scen_obj, goals):
  import time
  global waiting
  #print "new new new new new new"
  #pub = None #rospy.Publisher('/move_base/goal', MoveBaseActionGoal, queue_size=10)
  waiting = rospy.Subscriber('/move_base/result', MoveBaseActionResult, when_arrived, (scen_obj, goals))
  msg = create_goal_msg(goals[0],0)
  time.sleep(2)
  #apply_nav_vel_publisher()
  #pub.publish(msg)
  publish(msg)


def apply_nav_vel_publisher():
  rate = 4
  goal = "roslaunch robotican_demos attacker.launch rate:=%s" % (rate) 
  subprocess.Popen(goal, shell=True)
  print "/nev_val ATTACKING is started in rate:=%s" % (rate)

def publish(msg):
  goal = "rostopic pub /move_base/goal move_base_msgs/MoveBaseActionGoal " + msg
  subprocess.Popen(goal, shell=True)
  print "A Goal is Published" #: " + str(msg)
  
def apply_simulation(scen_obj, goals):
    print "simulation started..."
    publising_goals(scen_obj, goals)
