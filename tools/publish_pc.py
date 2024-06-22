import rclpy 
from rclpy.node import Node
import sensor_msgs.msg as sensor_msgs
import nav_msgs.msg as nav_msgs
import std_msgs.msg as std_msgs
import numpy as np
import open3d as o3d
import csv
import os
import math
import time 

import warnings

from ament_index_python.packages import get_package_share_directory

from evloc.gl_6dof import gl_6dof
from evloc.read_points import read_points
from evloc.common_classes import Color
from evloc.de_6dof import de_6dof
from evloc.pso_6dof import pso_6dof 
from evloc.iwo_6dof import iwo_6dof 
from evloc.generate_point_cloud import generate_point_cloud
from evloc.ask_params import ask_params

########### GLOBAL CONSTANTS ###########

PACKAGE_PATH = os.path.join(get_package_share_directory('evloc'), 'resources')

GROUNDTRUTH_FILE_PATH = f"{PACKAGE_PATH}/groundtruth_data.csv"
LOCAL_CLOUDS_FOLDER = f"{PACKAGE_PATH}/local_clouds"

DOWN_SAMPLING_FACTOR_GLOBAL = 0.004     # factor de downsampling para mapa, hay que reducir puntos en ambas nubes
DOWN_SAMPLING_FACTOR = 0.01             # factor de downsampling para scan
POP_RATIO = 0.01

########################################

# Filter out the RuntimeWarning for invalid value encountered in divide
# For when NaN is calculated in add_noise_to_pc.
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

class PCD(Node):

    def __init__(self):
        super().__init__('pcd_node')

        self.declare_parameter('auto', False)

        auto_color = Color.RED

        self.auto_mode = self.get_parameter('auto').value

        if self.auto_mode:
            auto_color = Color.GREEN

        print(auto_color + f"\nAuto Mode: {self.auto_mode}" + Color.END)

        self.pcd_publisher_local = self.create_publisher(sensor_msgs.PointCloud2, 'evloc_local', 10)
        self.pcd_publisher_global = self.create_publisher(sensor_msgs.PointCloud2, 'evloc_global', 10)

        self.auto_mode = False # CAMBIAR A MANO

    def run(self):
        id_cloud = 0
        while True:
            
            if not self.auto_mode:
                id_cloud = None
                selected = False
                # Local Cloud
                while not selected:
                    num_clouds = len(os.listdir(LOCAL_CLOUDS_FOLDER))

                    print(Color.BOLD + f'\nAvailable scans [1-{num_clouds}]' + Color.END)
                    id_cloud = input(Color.BOLD + "Select cloud as real scan: " + Color.END)
                    try:
                        if int(id_cloud) > num_clouds or int(id_cloud) < 1:
                            print(f'Error. Selected cloud ({id_cloud}) does not exist.') 
                        else:
                            id_cloud = int(id_cloud)
                            selected = True

                    except ValueError as e:
                        print(f'Error: Invalid Number. {e}')
            else:
                id_cloud = id_cloud + 1
                if id_cloud > 44:
                    id_cloud = 1
                    while True:
                        user_input = input("Restart? (y/n): ")
                        if user_input == 'y':
                            pass
                        elif user_input == 'n':
                            exit(1)
                        else:
                            print("Invalid answer. Please type 'y' for yes or 'n' for no.")

                time.sleep(1)

            print(Color.BOLD + "\n------------------------------------" + Color.END)

            # map_global_ori = o3d.io.read_point_cloud(f"{PACKAGE_PATH}/map_global_ori.ply")
            # map_global = map_global_ori.uniform_down_sample(every_k_points=int(1 / DOWN_SAMPLING_FACTOR_GLOBAL)) # Original PointCloud (Global Map)

            real_scan_ori = o3d.io.read_point_cloud(f"{PACKAGE_PATH}/local_clouds/cloud_{id_cloud}.ply")
            map_local = real_scan_ori.uniform_down_sample(every_k_points=int(1 / DOWN_SAMPLING_FACTOR))         # User Selected PointCloud (Local Map)

            downsample_2 = 1
            points2 = np.asarray(map_local.points)[::downsample_2] # Downsampling. Son demasiados puntos para RVIZ
            pcd_global = self.point_cloud(points2, 'map')
            self.pcd_publisher_global.publish(pcd_global)
            print(f"Global PointCloud with dimensions {points2.shape} has been published.")

            print(Color.BOLD + "\n------------------------------------" + Color.END)

    def ask_restart(self):
        while True:
            user_input = input("Restart? (y/n): ")
            if user_input == 'y':
                return True
            elif user_input == 'n':
                return False
            else:
                print("Invalid answer. Please type 'y' for yes or 'n' for no.")

    def point_cloud(self, points, parent_frame):
        """ Creates a point cloud message.
        Args:
            points: Nx3 array of xyz positions.
            parent_frame: frame in which the point cloud is defined
        Returns:
            sensor_msgs/PointCloud2 message

        Code source:
            https://gist.github.com/pgorczak/5c717baa44479fa064eb8d33ea4587e0

        References:
            http://docs.ros.org/melodic/api/sensor_msgs/html/msg/PointCloud2.html
            http://docs.ros.org/melodic/api/sensor_msgs/html/msg/PointField.html
            http://docs.ros.org/melodic/api/std_msgs/html/msg/Header.html

        """
        # In a PointCloud2 message, the point cloud is stored as an byte 
        # array. In order to unpack it, we also include some parameters 
        # which desribes the size of each individual point.
        ros_dtype = sensor_msgs.PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize # A 32-bit float takes 4 bytes.

        data = points.astype(dtype).tobytes() 

        # The fields specify what the bytes represents. The first 4 bytes 
        # represents the x-coordinate, the next 4 the y-coordinate, etc.
        fields = [sensor_msgs.PointField(
            name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
            for i, n in enumerate('xyz')]

        # The PointCloud2 message also has a header which specifies which 
        # coordinate frame it is represented in. 
        header = std_msgs.Header(frame_id=parent_frame)

        return sensor_msgs.PointCloud2(
            header=header,
            height=1, 
            width=points.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=(itemsize * 3), # Every point consists of three float32s.
            row_step=(itemsize * 3 * points.shape[0]),
            data=data
        )

def main(args=None):
    rclpy.init(args=args)
    pcd = PCD()
    pcd.run()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
