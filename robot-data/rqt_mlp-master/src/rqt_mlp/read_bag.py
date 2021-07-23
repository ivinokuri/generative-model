import rosbag
import rospy
import time

features_name = []

def get_features_nav_vel(bag, stime, etime):
    topic = '/cmd_vel'
    ret = []
    messages = list(bag.read_messages(start_time=rospy.Time.from_sec(stime), end_time=rospy.Time.from_sec(etime), topics=[topic]))
    vel_linear_x = __get_linear_velocity_x(messages)
    ret.append(vel_linear_x)
    vel_angular_z = __get_angular_velocity_z(messages)
    ret.append(vel_angular_z)
    return ret

def get_messages(first_time, bag, stime, etime):
    messages = list(bag.read_messages(start_time=rospy.Time.from_sec(stime), end_time=rospy.Time.from_sec(etime), topics=['/cmd_vel']))
    for _1,_2,_3 in messages:
        # print _2
        vel_z = _2.angular.z
        print vel_z

        # print _2
        # print _3
        print str(_3.to_sec()) + " - " + str(first_time) + " = " + str(_3.to_sec() - first_time)

        # print item[2]-genpy.Time.from_sec(stime)
    # print messages
    if messages:
        print "Counter " + str(len(messages))
        # print "Traffic " + str(__get_my_traffic(messages))
        # print "max_consecutive " + str(max([0] + __get_consecutive_times(messages)))
        # print "mean_consecutive " + str(__mean_consecutive_time_messages(messages))
        # print "var_consecutive " + str(__var_consecutive_time_messages(messages))
        # print "age " + str(__age_messages(messages))


def __get_bytes_size(data):
    import StringIO
    buff = StringIO.StringIO()
    data.serialize(buff)
    return len(buff.getvalue())


def __get_my_traffic(topic_messages):
    traffics = [__get_bytes_size(m) for _, m, _ in topic_messages]
    return sum(traffics)


def __to_seconds(t):
    # print t
    # print t.secs
    # print t.nsecs
    # print t.secs + t.nsecs * pow(10, -9)
    return t.secs + t.nsecs * pow(10, -9)


def __get_consecutive_times(messages):
    consecutive_times = []
    for (_, _, t1), (_,_,t2) in zip(messages[:-1], messages[1:]):
	t1 = t1.to_sec()# __to_seconds(t1)
	t2 = t2.to_sec()#__to_seconds(t2)
	consecutive_times.append(t2-t1)
    return consecutive_times


def __max_consecutive_time_messages(messages):
    consecutive_times = __get_consecutive_times(messages)
    return max([0] + consecutive_times)


def __mean_consecutive_time_messages(messages):
    import statistics as stat
    consecutive_times = __get_consecutive_times(messages)
    if len(consecutive_times) < 1:
      return 0;
    return stat.mean(consecutive_times)


def __var_consecutive_time_messages(messages):
    import statistics as stat
    consecutive_times = __get_consecutive_times(messages)
    if len(consecutive_times) < 2:
      return 0;
    return stat.variance(consecutive_times)


def __age_messages(messages):
    import statistics as stat
    NO_AGE_MEAN = 0
    ages = []
    for topic, m, t in messages:
      if m._has_header:
        send_time = __to_seconds(m.header.stamp) * pow(10,9)
        arrival_time = __to_seconds(t) * pow(10,9)
        ages.append(arrival_time - send_time)
      else:
        ages.append(0)
        if len(ages) > 0:
          return stat.mean(ages)
        else:
          return NO_AGE_MEAN

def __get_linear_velocity_x(messages):
    if len(messages) < 1:
      return 0
    topic, msg, t = messages[-1]
    vel_x = msg.linear.x
    return vel_x

def __get_angular_velocity_z(messages):
    if len(messages) < 1:
      return 0
    topic, msg, t = messages[-1]
    vel_z = msg.angular.z
    return vel_z

def __update_features_name(*names):
    for name in names:
      if name not in features_name:
        features_name.append(name)
    return features_name


def __get_joint_states(messages):
    def trans(x):
      inf = 10000000
      if x == float('Inf'):
        return inf
      if x == -float('Inf'):
        return -1 * inf
      if x.__str__() == 'nan':
        return inf * inf
      if x.__str__() == '-nan':
        return -1 * inf * inf
      if x == float('NaN') or x == float('naN') or x == float('Nan') or x == float('nan'):
        return inf * inf
      if x == -float('NaN') or x == -float('naN') or x ==  -float('Nan') or x == -float('nan'):
        return -1 * inf * inf
      return x
    names = ['head_pan_joint', 'head_tilt_joint', 'left_finger_joint', 'left_wheel_joint', 'right_finger_joint', 'right_wheel_joint', 'rotation1_joint', 'rotation2_joint', 'shoulder1_joint', 'shoulder2_joint', 'shoulder3_joint', 'torso_joint', 'wrist_joint']
    info = ["position", "effort", "velocity"]
    for inf in info:
      for name in names:
        __update_features_name("%s %s" % (name, inf))
    if len(messages) < 1:
      return [0] * (len(names) * len(info))
    topic, msg, t = messages[-1]
    ret = map(trans, list(msg.position + msg.effort + msg.velocity))
    return ret


def get_features_joints_info(bag, stime, etime):
    topic = '/joint_states'
    ret = []
    messages = list(bag.read_messages(start_time=rospy.Time.from_sec(stime), end_time=rospy.Time.from_sec(etime), topics=[topic]))
    ret = __get_joint_states(messages)
    return ret

print "####### big bug #######"
count = 0
bag = rosbag.Bag('/home/lab/bags/1_0.bag')

stime = bag.get_start_time()
first_time = bag.get_start_time()
# print stime
# print rospy.Time.from_sec(stime) # Create a new Time instance from a float seconds value
# print genpy.Time.from_sec(stime)
etime = bag.get_end_time()
# print etime
# print genpy.Time.from_sec(etime)

# print time.time()
# print rospy.Time.from_sec(time.time())
#
# print rospy.Time.from_sec(etime) - rospy.Time.from_sec(stime)
print (rospy.Time.from_sec(etime) - rospy.Time.from_sec(stime)).to_sec()

# print etime - stime

while (stime + 1 <= etime):
    get_messages(first_time, bag, stime, stime + 1)
    # print "velocity x,z " + str(get_features_nav_vel(bag, stime, stime + 1))
    # array = get_features_joints_info(bag, stime, stime + 1)
    # count = 0
    # for item in features_name:
    #     print str(item) + " " + str(array[count])
    #     count += 1
    stime = stime + 1

# for topic, msg, t in bag.read_messages(topics=['/cmd_vel']):
#     count += 1
#     print msg

bag.close()

print "####### small bug #######"
count1 = 0
bag1 = rosbag.Bag('/home/lab/sub.bag')
stime = bag1.get_start_time()
first_time = bag1.get_start_time()
# print stime
# print genpy.Time.from_sec(stime)
etime = bag1.get_end_time()
# print etime
# print genpy.Time.from_sec(etime)

print (rospy.Time.from_sec(etime) - rospy.Time.from_sec(stime)).to_sec()

# print etime - stime

while (stime + 1 <= etime):
    get_messages(first_time ,bag1, stime, stime + 1)
    # print "velocity x,z " + str(get_features_nav_vel(bag1, stime, stime + 1))
    # array = get_features_joints_info(bag1, stime, stime + 1)
    # count = 0
    # for item in features_name:
    #     print str(item) + " " + str(array[count])
    #     count += 1
    stime = stime + 1


# for topic1, msg1, t1 in bag1.read_messages(topics=['/cmd_vel']):
#     count1 += 1
#     print msg1
# print count
# print count1
bag1.close()





