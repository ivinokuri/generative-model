* before use this upgrade version code, over on the original basic code from this repository:

https://github.com/matansar/rqt_mlp - you can find explain about this code - on docs of google (https://docs.google.com) - in this account (plpbgu@post.bgu.ac.il) that write in hebrew - with this title: rqt Machine-Learning-Plugin (MLP)

# Instructions rqt Machine-Learning-Plugin (MLP) #
This repository offers a new rqt plugin that allows you to produce a dataset for machine learning. <br/>
#### In order to install the rqt plugin, MLP, please follow the following steps: ####
 * Install Ubuntu 14.04 64-bit
 * Install Indigo ROS distribution: http://wiki.ros.org/indigo/Installation/Ubuntu
 * Install and configure your ROS environment: http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment
 * Install RoboTiCan project: http://wiki.ros.org/robotican/Tutorials/Installation
 * Install python and then the following python packages:
    * Install pandas package:
      ```{r, engine='bash', count_lines}
      sudo apt-get install python-pip
      ```
      -- if you have dependence problems, run the following command to fix it and then run the above command:
      ```{r, engine='sh', count_lines}
      sudo apt-get -f install
      ```
      Then run:
      ```{r, engine='sh', count_lines}
      sudo pip install numpy
      sudo pip install pandas
      ```
    * Install statistics package:
         ```{r, engine='sh', count_lines}
      sudo pip install statistics
      ```
      
 * Install rosdiagnostic package: https://github.com/bguplp/rosDiagnostic.git
 
 * Download or clone this project using terminal/bash:
    ```{r, engine='sh', count_lines}
    cd ~/catkin_ws/src/
    git clone https://github.com/bguplp/rosDiagnostic.git
    ```
 
 * Run the setup.sh file and compile:
   ```{r, engine='sh', count_lines}
   cd ~/catkin_ws/src/rqt_mlp/install/
   chmod +x setup.sh
   ./setup.sh
   ```
 * Rebuild your project:
  ```{r, engine='sh', count_lines}
   cd ~/catkin_ws
   catkin_make
   ```
 * First time you run rqt, you need to run:
   ```{r, engine='sh', count_lines}
   cd ~/catkin_ws/
   rqt --force-discover
   ```
   You should run roscore before:
   ```{r, engine='sh', count_lines}
   roscore
   ```
   
# videos by numbers of files:
   
* 01 - explain and how make list of topics
* 02 - how to make list of topic - another video
* 03 - end part of making list of topics and explaining before rocording bag
* 04 - how to record good bags
* 05 - how to record bad bags
* 06 - generate features to csv from bags for good and bad bags
* 07 - generate features to csv from bags to neutral bags and do test offline with this tool -  (https://github.com/bguplp/thesis_offline)
   
