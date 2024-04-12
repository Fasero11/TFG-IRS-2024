# Evolutive Localization Implemented in ROS2

This project implements evolutive localization in ROS2.

**IMPORTANT: You need to add `global_map_ori.ply` to the `evloc/resources` folder. Since this file is too big, it has been uploaded as a .rar compressed file.**

## How to Run

1. Create a parent folder. Inside the folder, create a `src` subfolder.
2. Clone the repository inside the `src` folder.
3. Extract the `map_global_ori.ply` file from `map_global_ori.rar` and place it in  `evloc/resources`
4. From the parent folder (where `src` is located), run:

    ```bash
    colcon build --symlink-install
    ```

    Since we are using Python, which doesn't need to be compiled, we can modify the files and rerun the node without needing to rebuild thanks to the `--symlink-install` flag. When the build is finished, three new folders will have been created alongside `src`: `build`, `install`, and `log`.

5. Source the workspace:

    ```bash
    source install/setup.bash
    ```

6. Run the node:

    ```bash
    ros2 run evloc evloc_node
    ```

    Or in automatic mode:

    ```bash
    ros2 run evloc evloc_node --ros-args --param auto:=true
    ```

You can visualize the results in RViz. Just type `rviz2` in the terminal and open the configuration file. Alternatively, you can subscribe to the `/evloc_global` and `/evloc_local` topics.

Data will be saved in an `errordata.csv` file in your HOME directory. You can run `averages.py` for a better visualization of this data.
You can run `convergences.py` to see how many times each point cloud converges (Be sure to have the `errordata.csv` file in your HOME directory).


## Run in simulation

- Repository: [Multi-turtlebot3-Gazebo-ROS2](https://github.com/Taeyoung96/Multi-turtlebot3-Gazebo-ROS2)
  - Add `--volume=/dev/shm:/dev/shm` to the docker run command
  - Always execute `xhost +local:docker` before launching the container
  - Add `--net=host`, `--pid=host`, and `--ipc=host` to docker run command
  - To subscribe to the topics outside the container you need to have the same user ID (UID) on both the container and the outside shell (probably UID=0 root).
  - For teleop use `ros2 run teleop_twist_keyboard teleop_twist_keyboard`

- Repository: [lidarslam_ros2](https://github.com/rsasaki0109/lidarslam_ros2)


## Screenshots

![Screenshot 1](https://github.com/Fasero11/TFG-IRS-2024/assets/86266311/6ea2ade6-6c87-43a7-930a-0ff16330e3f0)

![Screenshot 2](https://github.com/Fasero11/TFG-IRS-2024/assets/86266311/c74ad795-a647-4fa1-a4d4-06d8ea117fd9)

## Dependencies (Versions used for design and testing)
- Ubuntu 22.04
- ROS2-Humble
- Python 3.10.12
- Pandas 2.2.0
- numpy 1.26.4
- open3d 0.18.0
- setuptools 58.2.0


## License

[License](LICENSE)
