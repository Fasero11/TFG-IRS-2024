# Evolutive Localization Implemented in ROS2

This project implements evolutive localization in ROS2.

**IMPORTANT: You need to add `global_map_ori.ply` to the `evloc/resources` folder. Since this file is too big, it has been uploaded as a .rar compressed file.**

## How to Run

1. Create a folder. Inside the folder, create a `src` subfolder.
2. Clone the repository inside the `src` folder.
3. From the initial folder (where `src` is located), run:

    ```bash
    colcon build --symlink-install
    ```

    Since we are using Python, which doesn't need to be compiled, we can modify the files and rerun the node without needing to rebuild thanks to the `--symlink-install` flag. When the build is finished, three new folders will have been created alongside `src`: `build`, `install`, and `log`.

4. Source the workspace:

    ```bash
    source install/setup.bash
    ```

5. Run the node:

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

## Screenshots

![Screenshot 1](https://github.com/Fasero11/TFG-IRS-2024/assets/86266311/6ea2ade6-6c87-43a7-930a-0ff16330e3f0)

![Screenshot 2](https://github.com/Fasero11/TFG-IRS-2024/assets/86266311/c74ad795-a647-4fa1-a4d4-06d8ea117fd9)

## License

[License](LICENSE)
