import rosgraph
import calc_bw
import subprocess
import rospy
import inspect
import os

ret = []

filepath = os.path.abspath(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/../") + "/topics"
print filepath
if os.path.isfile(filepath+"/topics.txt"):
    with open(filepath+"/topics.txt", 'r') as f:
        ret = f.read().splitlines()
        ret.append("/rosout_agg")
        ret.append("/rosout")
if os.path.isfile(filepath + "/empty_topics.txt"):
    with open(filepath + "/empty_topics.txt", 'r') as f:
        ret1 = f.read().splitlines()
        ret.extend(ret1)
if os.path.isfile(filepath + "/more_than_1MB.txt"):
    with open(filepath + "/more_than_1MB.txt", 'r') as f:
        ret2 = f.read().splitlines()
        ret.extend(ret2)

master = rosgraph.Master('topics_list')

topic_data_list4 = map(lambda l: l[0], master.getPublishedTopics(''))
topic_data_list4.sort()

for topic in topic_data_list4:
    if topic not in ret:
        is_larger_than_1MB = calc_bw.rostopicmain(["rostopic", "bw", topic])
        if (not is_larger_than_1MB) and is_larger_than_1MB != None:
            with open(filepath+"/topics.txt", "a") as f:
                f.write(topic+"\n")
                print topic
                f.close()
        elif is_larger_than_1MB and is_larger_than_1MB != None:
            with open(filepath+"/more_than_1MB.txt", "a") as f:
                f.write(topic+"\n")
                print topic
                f.close()
        else:
            with open(filepath+"/empty_topics.txt", "a") as f:
                f.write(topic+"\n")
                print topic
                f.close()

print "Try again to collect new topics"
rospy.sleep(2)
filepath1 = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
create_topic_list = "python " + filepath1 + "/create_topic_list.py"
subprocess.Popen(create_topic_list, shell=True)
