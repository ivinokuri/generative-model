cmake_minimum_required(VERSION 2.8.3)
project(rosdiagnostic)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

install(PROGRAMS scripts/rosdiagnostic
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
