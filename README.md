# TFG-IRS-2024
Evolutive Localization Implemented in ROS2

To enable automatic mode:
ros2 run evloc evloc_node --ros-args --param auto:=true

In this mode the program will loop with the default parameters.

To run in manual mode:
ros2 run evloc evloc_node 
or
ros2 run evloc evloc_node --ros-args --param auto:=false

Data will be saved in a .csv file in your HOME directory.
