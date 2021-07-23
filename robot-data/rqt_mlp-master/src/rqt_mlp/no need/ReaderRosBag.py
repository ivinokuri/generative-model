#!/usr/bin/env python
import rosbag
import rospy
import time
import pandas as pd


options = None
features_name = []
topics = ['/nav_vel', '/move_base/feedback']
window = 2

def update_features_name(*names):
  for name in names:
    if name not in features_name:
      features_name.append(name)

def default_values(value, amount):
  ret = [value]
  return ret * amount

def to_seconds(t):
  return t.secs + t.nsecs * pow(10,-9)

def is_include_nodes(node):
  exclude_nodes = ['gazebo','rviz','record', 'rqt', 'rosprofiler', 'rosgrapher']
  for ex_node in exclude_nodes:
    if ex_node in node:
      return False;
  return True;

##--------------------------------------------------------------------------------------------------------------------------------------------

def get_consecutive_times(messages):
  consecutive_times = []
  for (_, _, t1), (_,_,t2) in zip(messages[:-1], messages[1:]):
      t1 = to_seconds(t1)
      t2 = to_seconds(t2)
      consecutive_times.append(t2-t1)
  return consecutive_times

#feature 1
def count_messages(topic, messages):
  update_features_name("Counter(%s)" % topic)
  return len(messages)

#feature 2
def min_consecutive_time_messages(topic, messages):
  update_features_name("Min_Consecutive(%s)" % topic)
  consecutive_times = get_consecutive_times(messages)
  return min(consecutive_times)

#feature 3
def max_consecutive_time_messages(topic, messages):
  update_features_name("Max_Consecutive(%s)" % topic)
  consecutive_times = get_consecutive_times(messages)
  return max(consecutive_times)

#feature 4
def mean_consecutive_time_messages(topic, messages):
  import statistics as stat
  update_features_name("Mean_Consecutive(%s)" % topic)
  consecutive_times = get_consecutive_times(messages)
  return stat.mean(consecutive_times)

#feature 5
def var_consecutive_time_messages(topic, messages):
  import statistics as stat
  update_features_name("Var_Consecutive(%s)" % topic)
  consecutive_times = get_consecutive_times(messages)
  return stat.variance(consecutive_times)


##--------------------------------------------------------------------------------------------------------------------------------------------


#feature 10
def get_location_x(messages):
  update_features_name("location x")
  topic, msg, t = messages[-1]
  x = msg.feedback.base_position.pose.position.x
  return x
  
#feature 11
def get_location_y(messages):
  update_features_name("location y")
  topic, msg, t = messages[-1]
  y = msg.feedback.base_position.pose.position.y
  return y


##--------------------------------------------------------------------------------------------------------------------------------------------


#feature 20
def get_linear_velocity_x(messages):
  update_features_name("linear velocity x")
  topic, msg, t = messages[-1]
  vel_x = msg.linear.x
  return vel_x

#feature 21
def get_angular_velocity_z(messages):
  update_features_name("angular velocity z")
  topic, msg, t = messages[-1]
  vel_z = msg.angular.z
  return vel_z


##--------------------------------------------------------------------------------------------------------------------------------------------

def get_relevant_nodes(messages):
  messages = filter(lambda (n,m,t): is_include_nodes(m.node), messages) #related just to relevant nodes
  ret = []
  nodes = []
  for message in reversed(messages):
    (topic , m, t) = message
    node = m.node
    if node not in nodes:
      ret.append(message)
      nodes.append(node)
  return ret    

#feature 30
def get_nodes_count_threads(messages):
  import statistics as stat
  update_features_name("System Threads")
  threads = [m.threads for _, m, _ in messages]
  return sum(threads)

#feature 31
def get_nodes_max_mean_std_cpu_load(messages):
  import statistics as stat
  update_features_name("Nodes Max CPU", "Nodes Mean CPU", "Nodes Std CPU")
  cpu_maxs = [m.cpu_load_max for _, m, _ in messages]
  cpu_means = [m.cpu_load_mean for _, m, _ in messages]
  cpu_stds = [m.cpu_load_std for _, m, _ in messages]
  return [max(cpu_maxs), stat.mean(cpu_means), stat.mean(cpu_stds)]

#feature 32
def get_nodes_max_mean_std_virt_mem(messages):
  import statistics as stat
  update_features_name("Nodes Max Virt-mem", "Nodes Mean Virt-mem", "Nodes Std Virt-mem")
  virl_mem_maxs = [m.virt_mem_max for _, m, _ in messages]
  virl_mem_means = [m.virt_mem_mean for _, m, _ in messages]
  virl_mem_stds = [m.virt_mem_std for _, m, _ in messages]
  return [max(virl_mem_maxs), stat.mean(virl_mem_means), stat.mean(virl_mem_stds)]

#feature 33
def get_nodes_max_mean_std_real_mem(messages):
  import statistics as stat
  update_features_name("Nodes Max Real-mem", "Nodes Mean Real-mem", "Nodes Std Real-mem")
  real_mem_maxs = [m.real_mem_max for _, m, _ in messages]
  real_mem_means = [m.real_mem_mean for _, m, _ in messages]
  real_mem_stds = [m.real_mem_std for _, m, _ in messages]
  return [max(real_mem_maxs), stat.mean(real_mem_means), stat.mean(real_mem_stds)]


##--------------------------------------------------------------------------------------------------------------------------------------------


#feature 40
def get_hosts_max_mean_std_cpu_load(messages):
  cpu_counter = len(messages[0][1].cpu_load_max)
  for stat in ["Max", "Mean", "Std"]:
    for cpu in range(1,cpu_counter + 1):
      update_features_name("%s CPU %s" % (stat, cpu))
  _, m, _ = messages[-1]
  cpu_maxs = list(m.cpu_load_max)
  cpu_means = list(m.cpu_load_mean)
  cpu_stds = list(m.cpu_load_std)  
  return cpu_maxs + cpu_means + cpu_stds

#feature 41
def get_hosts_mem_avail_use(messages):
  update_features_name("Hosts Max Use-mem", "Hosts Mean Use-mem", "Hosts Std Use-mem", "Hosts Max Avail-mem", "Hosts Mean Avail-mem", "Hosts Std Avail-mem")
  _, m, _ = messages[-1]
  return [m.phymem_used_max, m.phymem_used_mean, m.phymem_used_std, m.phymem_avail_max, m.phymem_avail_mean, m.phymem_avail_std]


##--------------------------------------------------------------------------------------------------------------------------------------------


#feature 50
def count_active_connections(topic, topic_messages):
  update_features_name('%s Active Connections' % topic)
  connections = []
  counter = 0
  for _, m, _ in topic_messages:
    connection = (m.node_pub, m.node_sub)
    if connection not in connections:
      connections.append(connection)
      counter = counter + 1
  return counter

#feature 51
def get_dropped_msgs(topic, topic_messages):
  update_features_name('%s Dropped' % topic)
  dropped_msgs = reduce(lambda acc,(n, m, t): acc + m.dropped_msgs, topic_messages, 0)
  return dropped_msgs

#feature 52
def get_delivered_msgs(topic, topic_messages):
  update_features_name('%s Delivered' % topic)
  delivered_msgs = reduce(lambda acc,(n, m, t): acc + m.delivered_msgs, topic_messages, 0)
  return delivered_msgs 

#feature 53
def get_traffic(topic, topic_messages):
  update_features_name("%s Traffic(Bytes)" % topic)
  traffics = [m.traffic for _, m, _ in topic_messages]
  return sum(traffics)

#feature 54
def get_max_mean_std_period(topic, topic_messages):
  update_features_name("%s Max Period" % topic, "%s Mean Period" % topic, "%s Std Period" % topic)
  if len(topic_messages) == 0:
    return default_values(0,3)
  _, m, _ = topic_messages[-1]
  return [to_seconds(m.period_max), to_seconds(m.period_mean), to_seconds(m.period_stddev)]
  
#feature 55
def get_max_mean_std_age(topic, topic_messages):
  update_features_name("%s Max Age" % topic, "%s Mean Age" % topic, "%s Std Age" % topic)
  if len(topic_messages) == 0:
    return default_values(0,3)
  _, m, _ = topic_messages[-1]
  return [to_seconds(m.stamp_age_max), to_seconds(m.stamp_age_mean), to_seconds(m.stamp_age_stddev)]


##--------------------------------------------------------------------------------------------------------------------------------------------


# General type - return the features in the same calling's order
def get_general_features(bag, stime, etime):
  ret = []
  for topic in topics:
    
    messages = list(bag.read_messages(start_time = rospy.Time.from_sec(stime), end_time=rospy.Time.from_sec(etime), topics = [topic]))
    counter = count_messages(topic, messages)
    min_consecutive_messages = min_consecutive_time_messages(topic, messages)
    max_consecutive_messages = max_consecutive_time_messages(topic, messages)
    mean_consecutive_messages = mean_consecutive_time_messages(topic, messages)
    var_consecutive_messages = var_consecutive_time_messages(topic, messages)
    ret.extend([counter, min_consecutive_messages, max_consecutive_messages, mean_consecutive_messages, var_consecutive_messages])
  return ret

# Host_statistics - return the features in the same calling's order
def get_features_host_statistics(bag, stime, etime):
  topic = '/host_statistics'
  messages = list(bag.read_messages(start_time = rospy.Time.from_sec(stime), end_time=rospy.Time.from_sec(etime), topics = [topic]))
  cpu_max_mean_std = get_hosts_max_mean_std_cpu_load(messages)
  mem_max_mean_std = get_hosts_mem_avail_use(messages)
  return cpu_max_mean_std + mem_max_mean_std

# Node_statistics - return the features in the same calling's order
def get_features_node_statistics(bag, stime, etime):
  topic = '/node_statistics'
  messages = list(bag.read_messages(start_time = rospy.Time.from_sec(stime), end_time=rospy.Time.from_sec(etime), topics = [topic]))
  messages = get_relevant_nodes(messages)
  threads_counter = get_nodes_count_threads(messages)
  max_mean_std_cpu_load = get_nodes_max_mean_std_cpu_load(messages)
  max_mean_std_virt_mem = get_nodes_max_mean_std_virt_mem(messages)
  max_mean_std_real_mem = get_nodes_max_mean_std_real_mem(messages) 
  return  [threads_counter] + max_mean_std_cpu_load + max_mean_std_virt_mem + max_mean_std_real_mem
  
# Statistics - return the features in the same calling's order  
def get_features_statistics(bag, stime, etime):
  topic = '/statistics'
  messages = list(bag.read_messages(start_time = rospy.Time.from_sec(stime), end_time=rospy.Time.from_sec(etime), topics = [topic]))
  topic_statistics = []
  for topic in topics:
    topic_messages = filter(lambda (n ,m, t) : m.topic == topic , messages)
    active_connection_counter = count_active_connections(topic, topic_messages)
    dropped_msgs = get_dropped_msgs(topic, topic_messages)
    delivered_msgs = get_delivered_msgs(topic, topic_messages)
    traffic = get_traffic(topic, topic_messages)
    max_mean_std_period = get_max_mean_std_period(topic, topic_messages)
    max_mean_std_age = get_max_mean_std_age(topic, topic_messages)
    topic_statistics.extend([active_connection_counter, dropped_msgs, delivered_msgs, traffic] + max_mean_std_period + max_mean_std_age)
  return topic_statistics
  
# Nav_vel - return the features in the same calling's order
def get_features_nav_vel(bag, stime, etime):
  topic = '/nav_vel'
  messages = list(bag.read_messages(start_time = rospy.Time.from_sec(stime), end_time=rospy.Time.from_sec(etime), topics = [topic]))
  vel_linear_x = get_linear_velocity_x(messages)
  vel_angular_z = get_angular_velocity_z(messages)
  return [vel_linear_x, vel_angular_z]
    
# Movebase_Feedback - return the features in the same calling's order    
def get_features_move_bsae_feedback(bag, stime, etime):
  topic = '/move_base/feedback'
  messages = list(bag.read_messages(start_time = rospy.Time.from_sec(stime), end_time=rospy.Time.from_sec(etime), topics = [topic]))
  location_x = get_location_x(messages)
  location_y = get_location_y(messages)
  return [location_x, location_y]

# return the features in the same calling's order
def get_features(bag, stime, etime):    
  general_features = get_general_features(bag, stime, etime)
  nav_vel_features = get_features_nav_vel(bag, stime, etime)
  mb_feedback_features = get_features_move_bsae_feedback(bag, stime, etime)
  node_statistics_features = get_features_node_statistics(bag, stime, etime)
  host_statistics_features = get_features_host_statistics(bag, stime, etime)
  statistics_features = get_features_statistics(bag, stime, etime)
  return general_features + nav_vel_features + mb_feedback_features + node_statistics_features + host_statistics_features + statistics_features


  
def read_bag(bag_file):
  df = pd.DataFrame()
  bag = rosbag.Bag(bag_file)
  stime = bag.get_start_time() 	# represented by float
  etime = bag.get_end_time()	# represented by float
  while (stime + window <= etime):
    print("stime = %s | etime = %s" % (stime , stime + window))
    #test(bag, stime, stime + window)
    features = get_features(bag, stime, stime + window)   
    df = df.append([features])
    stime = stime + 1
  bag.close()
  return df
  

def write_to_csv(df, csv_file):
  df.to_csv(csv_file, index=False, header=features_name)     

def ApplyOptions():
    from optparse import OptionParser
    global options
    parser = OptionParser()
    parser.add_option("-b", "--bagfile", dest="bag_file",help="run BAG file (.bag)", metavar="BAG")
    parser.add_option("-o", "--output", dest="output",help="write report to OUT (.csv)", metavar="OUT")
    #parser.add_option("-p", "--prefix", dest="prefix", help="PR of OUT", default='/home/matansar/catkin_ws/src/robotican/robotican_demos/src/thesis-robotics/RosBagTraining/GoForwordRes/', metavar="PR")
    options, args  = parser.parse_args()
    if not options.bag_file:
	parser.error('BAG file is not given')
    if not options.output:
	parser.error('OUT file is not given')


if __name__ == '__main__':
  ApplyOptions()
  df = read_bag(options.bag_file) 
  write_to_csv(df, options.output)
  
  
  