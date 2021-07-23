This directory contains several implemented anomalies:

1. move_base - this directory contains an anomaly at publishing velocity commands in move_base component. This anomaly affects message publication, dropping some messages. We have changed the file: move_base-publishing_0.4/src/move_base.cpp

2. laser - this directory contains an anomaly at laser scan. We simulate a fault in the robot's LIDAR, corrupting some of the fields in the distance vector returned.
We have changed the file: gazebo_ros_pkgs-laser_problem/gazebo_plugins/src/gazebo_ros_laser.cpp

3. velocity attack - this directory contains an anomaly at velocity messages. We add an attacker that publishes velocity messages to the relevant ROS topic, 
making it more difficult for the robot to reach its destination.