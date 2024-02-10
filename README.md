# TFG-IRS-2024
Evolutive Localization Implemented in ROS2.

How to run:

1) Create a folder. Inside the folder create a src subfolder.
2) Clone the repo inside the src folder
3) From the initial folder (where src is located) do "colcon build --symlink-install"
   Since we are using Python, which doesn't need to compile, we can modify the files and rerun the node without needing to rebuild thanks to the "--symlink-install" flag
   When the build is finished, three new folders will have been created alongside "src": build, install, and log
4) Source the workspace. "source install/setup.bash"
5) Run the node: ros2 run evloc evloc_node
   Or in automatic mode: ros2 run evloc evloc_node --ros-args --param auto:=true

You can visualize the results in rviz. Just type rviz2 in the terminal and open the configuration file. Or just subscribe to the /evloc_global and /evloc_local topics.
   
Data will be saved in an errordata.csv file in your HOME directory.
You can run averages.py for a better visualization of this data.

![Screenshot from 2024-02-09 18-03-16](https://github.com/Fasero11/TFG-IRS-2024/assets/86266311/6ea2ade6-6c87-43a7-930a-0ff16330e3f0)

![Screenshot from 2024-02-09 18-02-30](https://github.com/Fasero11/TFG-IRS-2024/assets/86266311/c74ad795-a647-4fa1-a4d4-06d8ea117fd9)
