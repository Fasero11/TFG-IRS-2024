import open3d as o3d
from evloc.common_functions import spatial_rotation

# ANSI color escape codes
class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class Solution:
    """
    Class where the solution of the algorithm is stored.
    it: Number of iterations to converge.
    timediff: Time elapsed during the execution.
    estimate: Estimated pose
    pos_error: Position error in meters
    ori_error: Orientation error in degrees.
    map_global: Global map
    real_scan: Local map
    """
    def __init__(self, it, timediff, estimate, pos_error, ori_error, map_global, real_scan, stop_condition):
        self.it = it
        self.time = timediff
        self.pose_estimate = estimate
        self.pos_error = pos_error
        self.ori_error = ori_error
        self.map = map_global
        self.loc_scan = self.get_loc_scan(real_scan, estimate)
        self.stop_condition = stop_condition

    def get_loc_scan(self, real_scan, estimate):
        new_points = spatial_rotation(real_scan, estimate)

        new_pc = o3d.geometry.PointCloud()
        new_pc.points = o3d.utility.Vector3dVector(new_points)

        return new_pc
    
class Algorithm:
    """
    Class that stores the parameters of the algorithm that will be used
    """
    def __init__(self, type=None, NPini=None, iter_max=None, D=None, F=None, CR=None, w=None, wdamp=None, c1=None, c2=None,
                 Smin=None, Smax=None, exponent=None, sigma_initial=None, sigma_final=None):
        self.type = type
        self.NPini = NPini
        self.iter_max = iter_max
        self.D = D
        self.F = F
        self.CR = CR
        self.w = w
        self.wdamp = wdamp
        self.c1 = c1
        self.c2 = c2
        self.Smin = Smin 
        self.Smax = Smax
        self.exponent = exponent
        self.sigma_initial = sigma_initial 
        self.sigma_final = sigma_final