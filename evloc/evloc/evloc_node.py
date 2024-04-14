import rclpy 
from rclpy.node import Node
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
import numpy as np
import open3d as o3d
import csv
import os

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

##############################################################

def get_groundtruth_data(GROUNDTRUTH_FILE_PATH, id_cloud):
    """
    Reads the row "id_cloud" from the GROUNDTRUTH_FILE_PATH and returns it.
    """
    try:
        with open(GROUNDTRUTH_FILE_PATH, 'r') as file:
            csv_reader = csv.reader(file)
            
            for _ in range(int(id_cloud)):
                next(csv_reader)
            
            groundtruth_str = next(csv_reader)
            groundtruth = np.array(groundtruth_str, dtype=float)

            return groundtruth
        
    except FileNotFoundError:
        print("El archivo CSV no fue encontrado.")
    except StopIteration:
        print("La fila especificada excede el número de filas en el archivo CSV.")

############################################################
########################## MAIN ############################
############################################################

class PCD(Node):

    def __init__(self):
        super().__init__('pcd_node')
        
        # Declara el parámetro mi_parametro con un valor predeterminado
        self.declare_parameter('auto', False)
        self.declare_parameter('simulated', False)

        auto_color = Color.RED
        simulated_color = Color.RED

        self.auto_mode = self.get_parameter('auto').value
        self.simulated = self.get_parameter('simulated').value

        if self.auto_mode:
            auto_color = Color.GREEN

        if self.simulated:
            simulated_color = Color.GREEN

        print(auto_color + f"\nAuto Mode: {self.auto_mode}" + Color.END)
        print(simulated_color + f"Simulated: {self.simulated}" + Color.END)

        self.pcd_subscriber = self.create_subscription(
            sensor_msgs.PointCloud2,
            '/velodyne_points',
            self.listener_callback,
            10  # el número de mensajes en la cola
        )

        self.pcd_publisher_local = self.create_publisher(sensor_msgs.PointCloud2, 'evloc_local', 10)
        self.pcd_publisher_global = self.create_publisher(sensor_msgs.PointCloud2, 'evloc_global', 10)
        self.cloud_points = None

    def listener_callback(self, msg):
        self.cloud_points = msg

    def run(self):
        # Ask once before starting if in auto mode.
        if self.auto_mode:
            id_cloud, err_dis, unif_noise, algorithm_type, version_fitness, user_NPini, user_iter_max = ask_params(self.simulated)
        while True:

            print(Color.BOLD + "\n------------------------------------" + Color.END)

            # Ask every iteration if not in auto mode.
            if not self.auto_mode:
                id_cloud, err_dis, unif_noise, algorithm_type, version_fitness, user_NPini, user_iter_max = ask_params(self.simulated)

            map_global = None
            map_local = None
            real_groundtruth = None

            if (self.simulated):
                rclpy.spin_once(self) # Read once from subscribed topics
                while (self.cloud_points == None):
                    print("Waiting for local scan...")
                    rclpy.spin_once(self)
                
                map_global = o3d.io.read_point_cloud(f"{PACKAGE_PATH}/map_global_sim.pcd")
                real_groundtruth = np.full((6,), -1) # TODO: Get real robot pose

                # Transform map_local datatype
                points = read_points(self.cloud_points, skip_nans=True, field_names=("x", "y", "z"))
                point_list = np.array(list(points))
                map_local = o3d.geometry.PointCloud()
                map_local.points = o3d.utility.Vector3dVector(point_list)
            else:
                map_global_ori = o3d.io.read_point_cloud(f"{PACKAGE_PATH}/map_global_ori.ply")
                real_scan_ori = o3d.io.read_point_cloud(f"{PACKAGE_PATH}/local_clouds/cloud_{id_cloud}.ply")
                map_global = map_global_ori.uniform_down_sample(every_k_points=int(1 / DOWN_SAMPLING_FACTOR_GLOBAL)) # Original PointCloud (Global Map)
                map_local = real_scan_ori.uniform_down_sample(every_k_points=int(1 / DOWN_SAMPLING_FACTOR))         # User Selected PointCloud (Local Map)
                real_groundtruth = get_groundtruth_data(GROUNDTRUTH_FILE_PATH, id_cloud)              

            #TEST MAP PUBLISHING
            # points = np.asarray(map_global.points) # Downsampling. Son demasiados puntos para RVIZ
            # pcd_global = self.point_cloud(points, 'map')
            # self.pcd_publisher_global.publish(pcd_global)
            # print(f"Global PointCloud with dimensions {points.shape} has been published.")

            # points2 = np.asarray(map_local.points) # Downsampling. Son demasiados puntos para RVIZ
            # pcd_local = self.point_cloud(points2, 'map')
            # self.pcd_publisher_local.publish(pcd_local)
            # print(f"Local PointCloud with dimensions {points2.shape} has been published.")

            print(f"\n\nObtained global scan with dimensions {np.asarray(map_global.points).shape}")
            print(f"Obtained local scan with dimensions {np.asarray(map_local.points).shape}\n")

            points = generate_point_cloud(auto=self.auto_mode,
                                          id_cloud = id_cloud,
                                          err_dis = err_dis, 
                                          unif_noise = unif_noise,
                                          algorithm_type = algorithm_type,
                                          version_fitness = version_fitness,
                                          user_NPini = user_NPini,
                                          user_iter_max = user_iter_max,
                                          map_global = map_global,
                                          real_scan = map_local,
                                          groundtruth = real_groundtruth)

            if points is None:
                print("Error generating point cloud.")
                break
            
            ds_1 = 1
            ds_2 = 1
            if not self.simulated:
                ds_1 = 2
                ds_2 = 40
            
            self.publish_point_clouds(points, 'map', map_global, ds_1, ds_2)

            if not self.auto_mode:
                restart = self.ask_restart()
                if not restart:
                    self.destroy_node()  # Cierra el nodo antes de salir del bucle
                    break
            else:
                if not self.simulated:
                    # Loop for every cloud when in auto mode
                    id_cloud += 1
                    if id_cloud > 44:
                        id_cloud = 1

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


    def publish_point_clouds(self, points, parent_frame, global_map, downsample_1, downsample_2):
        
        points = points[::downsample_1] # Downsampling. Son demasiados puntos para RVIZ
        pcd = self.point_cloud(points, parent_frame)
        self.pcd_publisher_local.publish(pcd)
        print(f"Local PointCloud with dimensions {points.shape} has been published.")

        points2 = np.asarray(global_map.points)[::downsample_2] # Downsampling. Son demasiados puntos para RVIZ
        pcd_global = self.point_cloud(points2, parent_frame)
        self.pcd_publisher_global.publish(pcd_global)
        print(f"Global PointCloud with dimensions {points2.shape} has been published.")

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
