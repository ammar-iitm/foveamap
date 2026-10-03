cmake_minimum_required(VERSION 3.8)
project(foveamap_ros)

# Colcon/ament build for the FoveaMap ROS 2 adapter.
# The Python adapter itself has no ROS 2 dependency at import time; the
# build only wires message generation and installs the pure-Python package.

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FoveaCell.msg"
  "msg/GridTile.msg"
  "msg/LabeledPoints.msg"
  "msg/FoveaMapMetrics.msg"
  DEPENDENCIES std_msgs sensor_msgs
)

ament_python_install_package(${PROJECT_NAME})

ament_package()
