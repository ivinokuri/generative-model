import rosgraph
import rostopic
import rospy

master = rosgraph.Master('topics_list')

topic_data_list4 = map(lambda l: l[0], master.getPublishedTopics(''))
topic_data_list4.sort()

# with open("topics.txt", "w") as f:
for item in topic_data_list4:
    r = rostopic.ROSTopicHz(-1)
    s = rospy.Subscriber(item, rospy.AnyMsg, r.callback_hz)
    rospy.sleep(1)
    r.print_bw()
    print "refael"
    s.unregister()
