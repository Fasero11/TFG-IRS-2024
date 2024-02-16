import rclpy 
from rclpy.node import Node
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
import numpy as np
import open3d as o3d

from ament_index_python.packages import get_package_share_directory

import csv
import os
import time
from math import pi

import warnings

############################################################
############################################################
############################################################

# Filter out the RuntimeWarning for invalid value encountered in divide
# For when NaN is calculated in add_noise_to_pc.
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

PACKAGE_PATH = os.path.join(get_package_share_directory('evloc'), 'resources')

GROUNDTRUTH_FILE_PATH = f"{PACKAGE_PATH}/groundtruth_data.csv"
LOCAL_CLOUDS_FOLDER = f"{PACKAGE_PATH}/local_clouds"

DOWN_SAMPLING_FACTOR_GLOBAL = 0.004     # factor de downsampling para mapa, hay que reducir puntos en ambas nubes
DOWN_SAMPLING_FACTOR = 0.01             # factor de downsampling para scan
POP_RATIO = 0.01


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

class Population:
    """
    An agent of the algorithm.
    Includes a position and cost.
    """
    def __init__(self, position=[], cost=[]):
        self.Position = position
        self.Cost = cost
        
    def clone(self):
        # Crea una nueva instancia de la clase Population con los mismos valores
        return Population(self.Position.copy(), self.Cost.copy())

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
    def __init__(self, it, timediff, estimate, pos_error, ori_error, map_global, real_scan):
        self.it = it
        self.time = timediff
        self.pose_estimate = estimate
        self.pos_error = pos_error
        self.ori_error = ori_error
        self.map = map_global
        self.loc_scan = self.get_loc_scan(real_scan, estimate)

    def get_loc_scan(self, real_scan, estimate):
        new_points = spatial_rotation(real_scan, estimate)

        new_pc = o3d.geometry.PointCloud()
        new_pc.points = o3d.utility.Vector3dVector(new_points)

        return new_pc

class Algorithm:
    """
    Class that stores the parameters of the algorithm that will be used
    """
    def __init__(self, type=None, NPini=None, iter_max=None, D=None, F=None, CR=None):
        self.type = type
        self.NPini = NPini
        self.iter_max = iter_max
        self.D = D
        self.F = F
        self.CR = CR


def de_6dof(scanCloud,mapCloud,mapmax,mapmin,err_dis,NPini,D,iter_max,F,CR,version_fitness):
    """
    Differential Evolution with Thresholding and Discarding 
    Evolucion por mutación. En cada iteración, cada candidato (xi) genera uno
    nuevo (x(i+1)). Este nuevo es una combinación parámetro a parámetro del candidato
    antiguo y de una combinación tal que x(i+1)=F*(xc-xb)+xa, de otros 3 candidados de la poblacion xa,
    xb y xc escogidos aleatoriamente.
    El factor de mutación F determina como de "lejos" puede terminar cada nuevo parámetro en caso de mutar
    La tasa de cruce CR define qué porcentaje de parámetros de x(i+1) son mutados o se heredan
    de xi.
    """
    ##  Boundaries
    higherBoundX = mapmax[0]  # X Translation in meters
    lowerBoundX = mapmin[0]
    higherBoundY = mapmax[1]  # Y Translation in meters
    lowerBoundY = mapmin[1]
    higherBoundZ = mapmax[2]  # Z Translation in meters
    lowerBoundZ = mapmin[2]
    higherAngle_rx = mapmax[3]  # Rotation around X
    lowerAngle_rx = mapmin[3]
    higherAngle_ry = mapmax[4]  # Rotation around Y
    lowerAngle_ry = mapmin[4]
    higherAngle_rz = mapmax[5]  # Rotation around Z
    lowerAngle_rz = mapmin[5]

    # Problem Definition

    nVar = D            # Number of Decision Variables
    VarSize = [1, nVar]   # Size of Decision Variables Matrix
    minIt = 50  # Minimum number of iterations

    # DE Parameters
    # F - Mutation
    # CR - Crossover rate

    # Initialization
    # Empty Candidate Structure
    empty_population = Population()

    # Initialize Population
    population = [empty_population.clone() for _ in range(NPini)]
    rndmember = np.zeros(6)
    count = 0
    vis1 = 1
    vis2 = 1
    best_particle_cost = 100000000
    worst_particle_cost = 100000
    count_bestfix = 0  # Counters for algorithm convergence
    count_worsefix = 0
    count_avgfix = 0
    ind_reparto_error = 100000
    NP = NPini

    ########## LOOP 1 ###########
    for current_iteration in range(NPini):
        # Initialize Position
        if current_iteration == 0:  # first population is a vector of zeros
            population[current_iteration].Position = np.zeros(6)
        else:
            rndmember = np.zeros(6)
            for n in range(nVar):
                if n == 0:  # Translation
                    rndmember[n] = np.random.uniform(lowerBoundX, higherBoundX)
                elif n == 1:
                    rndmember[n] = np.random.uniform(lowerBoundY, higherBoundY)
                elif n == 2:
                    rndmember[n] = np.random.uniform(lowerBoundZ, higherBoundZ)
                elif n == 3:  # Angle
                    rndmember[n] = np.random.uniform(lowerAngle_rx, higherAngle_rx)
                elif n == 4:
                    rndmember[n] = np.random.uniform(lowerAngle_ry, higherAngle_ry)
                elif n == 5:
                    rndmember[n] = np.random.uniform(lowerAngle_rz, higherAngle_rz)

            population[current_iteration].Position = rndmember

        # Transform local cloud into each candidate's location
        cand_scan = o3d.geometry.PointCloud()
        cand_scan.points = o3d.utility.Vector3dVector(spatial_rotation(scanCloud.points, population[current_iteration].Position))

        # Cortamos el mapa global a los limites de la nube local, para comparar con menos puntos(Puede salir nube vacía,
        # en tal caso crear nube de ceros)
        aabb = cand_scan.get_axis_aligned_bounding_box()
        cand_scan_min_bound = aabb.get_min_bound()
        cand_scan_max_bound = aabb.get_max_bound()

        cotout_region = o3d.geometry.AxisAlignedBoundingBox(min_bound=(cand_scan_min_bound), max_bound=(cand_scan_max_bound))
        Mapa_3D_cut = mapCloud.crop(cotout_region)

        if Mapa_3D_cut.is_empty():
            Mapa_3D_cut = o3d.geometry.PointCloud()
            points_array = np.zeros((np.array(cand_scan.points).shape[0], 3))
            Mapa_3D_cut.points = o3d.utility.Vector3dVector(points_array)

        # Busqueda de NN para cada punto del scan colocado en la localización candidata
        kdtree = o3d.geometry.KDTreeFlann(Mapa_3D_cut)
        Idx = np.empty((1, len(cand_scan.points)))

        for i in range(len(cand_scan.points)):
            query = np.array(cand_scan.points[i])
            query = np.where(np.isnan(query), 0, query) # Replace NaN for 0
            # Realizar la búsqueda de los vecinos más cercanos
            knn_sol = kdtree.search_knn_vector_3d(query, 1)
            index = knn_sol[1][0]
            Idx[0][i] = index

        # Crear matriz de correspondencia
        points_array = np.asarray(Mapa_3D_cut.points)

        # Ensure 'Idx' contains integer values
        Idx = Idx.astype(int)

        # Create matrix of correspondence
        correspondence_mat = np.zeros((Idx.shape[1], 3))
        for j in range(Idx.shape[1]):
            correspondence_mat[j, :] = points_array[Idx[0, j], :]

        # Calcular distancias euclídeas de cada punto del mapa y del scan al punto candidato
        dist_NNmap = distance_pc_to_point(correspondence_mat, population[current_iteration].Position)
        dist_scancand = distance_pc_to_point(cand_scan.points, population[current_iteration].Position)
        
        # Evaluar y asignar el error de las medidas (distancia euclídea o absoluta)
        population[current_iteration].Cost = costfunction3d(dist_scancand, dist_NNmap, version_fitness, err_dis)

    ###### END LOOP 1

    #.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#
    # Bucle principal del algoritmo DE  #
    #.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#
    for it in range(iter_max):
        start_time = time.time()
        for pop_id in range(NP-1):
            # Mutación y cruce
            a, b, c = np.random.randint(0, NP, size=3)
            newmember = Population()
            newmember.Position = [0,0,0,0,0,0]

            for j in range(nVar): # that mutation only takes place a CR% of times, if not, that parameter remains from the original candidate xi
                if np.random.rand() < CR:
                    newmember.Position[j] = population[c].Position[j] + F * (population[a].Position[j] - population[b].Position[j])
                else:
                    newmember.Position[j] = population[pop_id].Position[j]

            # Apply limits for new member
            newmember.Position[0] = max(newmember.Position[0], lowerBoundX)
            newmember.Position[0] = min(newmember.Position[0], higherBoundX)

            newmember.Position[1] = max(newmember.Position[1], lowerBoundY)
            newmember.Position[1] = min(newmember.Position[1], higherBoundY)

            newmember.Position[2] = max(newmember.Position[2], lowerBoundZ)
            newmember.Position[2] = min(newmember.Position[2], higherBoundZ)

            newmember.Position[3] = max(newmember.Position[3], lowerAngle_rx)
            newmember.Position[3] = min(newmember.Position[3], higherAngle_rx)

            newmember.Position[4] = max(newmember.Position[4], lowerAngle_ry)
            newmember.Position[4] = min(newmember.Position[4], higherAngle_ry)

            newmember.Position[5] = max(newmember.Position[5], lowerAngle_rz)
            newmember.Position[5] = min(newmember.Position[5], higherAngle_rz)

            # Evaluación nuevamente
            cand_scan = o3d.geometry.PointCloud()
            cand_scan.points = o3d.utility.Vector3dVector(spatial_rotation(scanCloud.points, newmember.Position))

            # Cortamos el mapa a los límites de la nube (puede salir nube vacía, poner a 000)
            aabb = cand_scan.get_axis_aligned_bounding_box()
            cand_scan_min_bound = aabb.get_min_bound()
            cand_scan_max_bound = aabb.get_max_bound()

            cotout_region = o3d.geometry.AxisAlignedBoundingBox(min_bound=(cand_scan_min_bound), max_bound=(cand_scan_max_bound))
            Mapa_3D_cut = mapCloud.crop(cotout_region)

            if Mapa_3D_cut.is_empty():
                Mapa_3D_cut = o3d.geometry.PointCloud()
                points_array = np.zeros((np.array(cand_scan.points).shape[0], 3))
                Mapa_3D_cut.points = o3d.utility.Vector3dVector(points_array)

            kdtree = o3d.geometry.KDTreeFlann(Mapa_3D_cut)
            Idx = np.empty((1, len(cand_scan.points)))

            for i in range(len(cand_scan.points)):
                query = np.array(cand_scan.points[i])
                query = np.where(np.isnan(query), 0, query) # Replace NaN for 0
                # Realizar la búsqueda de los vecinos más cercanos
                knn_sol = kdtree.search_knn_vector_3d(query, 1)
                index = knn_sol[1][0]
                Idx[0][i] = index

            # Crear matriz de correspondencia
            points_array = np.asarray(Mapa_3D_cut.points)

            # Ensure 'Idx' contains integer values
            Idx = Idx.astype(int)

            # Create matrix of correspondence
            correspondence_mat = np.zeros((Idx.shape[1], 3))
            for j in range(Idx.shape[1]):
                correspondence_mat[j, :] = points_array[Idx[0, j], :]

            # Calcular distancias euclídeas de cada punto del mapa y del scan al punto candidato
            dist_NNmap = distance_pc_to_point(correspondence_mat, newmember.Position)
            dist_scancand = distance_pc_to_point(cand_scan.points, newmember.Position)

            # Evaluar y asignar el error de las medidas (distancia euclídea o absoluta)
            newmember.Cost = costfunction3d(dist_scancand, dist_NNmap, version_fitness, err_dis)

            # Actualizar el miembro si mejora
            if newmember.Cost < population[pop_id].Cost * 0.98:  # con umbral, el nuevo miembro debe mejorar más del 2% (evita efecto de ruido)
                population[pop_id] = newmember

        # Discarding, substituting the worst candidates for some of the best
        # members (speeds up convergence)

        disc_range=0.9 # percentage to keep, in this case first 90%, last 10% will be replaced
        repl_range=0.2 # percentage to repalce with, in this case first 20%

        population.sort(key=lambda x: x.Cost)

        # Discard a portion of the population
        for i in range(NP, int(disc_range * NP), -1):
            population[i-1] = population[np.random.randint(0, repl_range*NP)]

        # Sort the population based on 'Cost'
        population = sorted(population, key=lambda obj: obj.Cost)

        BestSol  = population[0]
        WorstSol = population[NP-1]
        sumcosts=0

        # Calculate average cost
        for k in range(NP):
            pop = population[k]
            sumcosts += pop.Cost
        average_cost = sumcosts / NP

        # Best and worst costs
        best_particle_cost_now = BestSol.Cost
        worst_particle_cost_now = WorstSol.Cost

        if count == 10:
            print(f"\nIt: {it}, {Color.GREEN}Best: {round(best_particle_cost_now, 4)}{Color.END}, {Color.RED}Worse: {round(worst_particle_cost_now,4)}{Color.END}, {Color.YELLOW}Average: {round(average_cost,4)}{Color.END}, Best/measure: {round(best_particle_cost_now/NP,4)}, Worse/best: {round(worst_particle_cost_now/best_particle_cost_now,4)}, Avg/best: {round(average_cost/NP/best_particle_cost_now,4)} \n Position (x, y, z, alpha, beta, theta): [{round(BestSol.Position[0],4)}, {round(BestSol.Position[1],4)}, {round(BestSol.Position[2],4)}, {round(BestSol.Position[3],4)}, {round(BestSol.Position[4],4)}, {round(BestSol.Position[5],4)}]\n")
            count=0
        count=count+1
        end_time = time.time()
        
        #print(f'Count: {count} in {round(end_time-start_time, 2)} seconds') # DEBUG

        # Convergence indicators
        if best_particle_cost_now < best_particle_cost:
            count_worsefix = 0
            count_avgfix = 0
            count_bestfix = 0  # Yes, improvement from the previous iteration
        else:
            count_bestfix += 1  # No, increment the counter for non-improvement

        best_particle_cost = best_particle_cost_now

        # Check if the worst candidate has improved
        if worst_particle_cost_now > worst_particle_cost:
            count_worsefix = 0
            count_avgfix = 0
            count_bestfix = 0  # Yes, improvement in the worst candidate
        else:
            count_worsefix += 1  # No, increment the counter for non-improvement

        worst_particle_cost = worst_particle_cost_now
 
        ind_reparto_error_aux = sumcosts / (NP*best_particle_cost)

        if (ind_reparto_error_aux < ind_reparto_error): #Mejora la media?
            count_avgfix=0
            count_worsefix=0
            count_bestfix=0  #si
        else:
            count_avgfix= count_avgfix+1 #no

        ind_reparto_error = ind_reparto_error_aux

        # Modificacion de parámetros del algoritmo en caliente
        if (worst_particle_cost/best_particle_cost < 2.5) and (ind_reparto_error < 2): #Reducción del factor de mutación(lo lejos que se mueve un candidato) si converge un poco
            F = 0.7
            if (vis1 == 1):
                print(Color.CYAN + f'F reduced to 0.7' + Color.END)
                vis1 = 0
        if (worst_particle_cost/best_particle_cost < 1.5) and (ind_reparto_error < 1.25): # Reducción mayor si converge mucho, busqueda más cerca de las pos. actuales
            F = 0.3
            NP = int(NPini/5)
            if (vis2 == 1):
               print(Color.CYAN + f'F reduced to 0.3' + Color.END)
               vis2 = 0

        # Condiciones de convergencia (todos costes iguales||poblacion mejor,
        # media y peor muy parecida || poblacion estancada || máximo de iteraciones)
        if all(obj.Cost == best_particle_cost_now for obj in population) or \
                (worst_particle_cost / best_particle_cost < 1.15 and ind_reparto_error < 1.15 and it >= minIt) or \
                (count_bestfix > 10 and count_worsefix > 10 and count_avgfix > 10 and it >= minIt):
            
            if all(obj.Cost == best_particle_cost for obj in population):
                stringcondition = 'total convergence'
            elif worst_particle_cost / best_particle_cost < (1.15 + err_dis) and ind_reparto_error < (1.15 + err_dis):
                stringcondition = 'normal convergence'
            elif count_bestfix > 10 and count_worsefix > 10 and count_avgfix > 10:
                stringcondition = 'invariant convergence'
            
            print(f'\n{Color.CYAN}Population converged in: {it} iterations and condition: {stringcondition}{Color.END}')
            break
    ########################################################
    ########################################################
        
    BestMember = BestSol.Position

    rmse_array =  BestSol.Cost

    bestCost = BestSol.Cost

    pcAligned = o3d.geometry.PointCloud()
    pcAligned.points = o3d.utility.Vector3dVector(spatial_rotation(scanCloud.points, BestMember))

    return(pcAligned, BestMember, bestCost, rmse_array, it)

def add_noise_to_pc(cloud, err_dis, unif_noise):
    """
    Returns a point cloud with the noise added.
    """

    sensnoise_PCmat = cloud.points
    dist_3d = distance_pc_to_point(cloud.points, [0, 0, 0])
    err_m = np.random.randn(dist_3d.size)  # Generating random samples from a normal distribution
    err_level = dist_3d * err_dis
    new_dist_3d = dist_3d + (err_m * err_level)

    CONTAMINATION_LEVEL = unif_noise
    CORRECT_MEASUREMENTS = 1 - CONTAMINATION_LEVEL
    err_m2 = (np.random.rand(*new_dist_3d.shape) * 0.5) + 0.25
    err_cont = new_dist_3d * err_m2
    err_m3 = np.random.rand(*new_dist_3d.shape)
    err_m3 = (err_m3 <= CORRECT_MEASUREMENTS).astype(int)

    err_m4 = np.ones_like(err_m3) - err_m3
    new_dist_3d = err_m3 * new_dist_3d + err_m4 * err_cont  # The new scan is formed

    noise_mat = (new_dist_3d / dist_3d).T * sensnoise_PCmat

    noise_mat_no_nan = np.nan_to_num(noise_mat)
    # Create an Open3D PointCloud
    new_cloud = o3d.geometry.PointCloud()
    new_cloud.points = o3d.utility.Vector3dVector(noise_mat_no_nan)

    return new_cloud


def costfunction3d(dist_scan, dist_map, version_fitness, err_dis):
    """
    Descripción: función de ajuste que se optimiza mediante el filtro de localización global.
    Las mediciones láser de la exploración actual se comparan con las mediciones láser
    en el vecino más cercano sobre el mapa real para calcular un valor de costo
    para una estimación. Este valor de costo se minimizará para obtener la solución
    del problema de localización global.

    En el caso de considerar la norma L2, la contribución al
    error en cada medida es el cuadrado de la diferencia entre la medida estimada y la real,
    dividido por la varianza multiplicada por 2, para aplicar criterios de convergencia.
    """
    if version_fitness == 1:
        error = np.sum(np.sum(((dist_scan - dist_map)**2) / (1 + 2 * (err_dis * dist_scan)**2)))

    # En el caso de considerar la norma L1, la contribución al
    # error en cada medida es la diferencia entre la medida estimada y la real, en valor
    # absoluto, dividido por la desviación típica, para aplicar criterios de convergencia.
    if version_fitness == 2:
        error = np.sum(np.sum(np.abs(dist_scan - dist_map) / (1 + err_dis * dist_scan)))

    return error


def distance_pc_to_point(cloud_mat, point):
    """
    Distance from each point of a pointcloud to a point

    -cloud_mat: Open3D PointCloud object
    -point: (x y z)
    - dist=horizontal vector with each PCpoint distance to point
    """
    cloud_array = np.asarray(cloud_mat)

    dist_mat = np.zeros((1, cloud_array.shape[0]))

    for i in range(cloud_array.shape[0]):
        dist_mat[0, i] = np.sqrt((cloud_array[i, 0] - point[0]) ** 2 +
                                 (cloud_array[i, 1] - point[1]) ** 2 +
                                 (cloud_array[i, 2] - point[2]) ** 2)

    return dist_mat


def spatial_rotation(point, p):
    """
    spatialTransformation
    The point gets transform by multiplying by the coordinate frame of the
    first scan according to the parameters
    Inputs:
      point: Horizontal vector nx3
      p:  Vertical vector 6x1 (rad)
    Outputs:
      transformed: Horizontal vector nx3 with the point transformed in the
      space
    """

    cAlpha = np.cos(p[3])
    sAlpha = np.sin(p[3])
    cBeta = np.cos(p[4])
    sBeta = np.sin(p[4])
    cGamma = np.cos(p[5])
    sGamma = np.sin(p[5])

    rotation_matrix = np.array([
        [cBeta * cGamma, -cBeta * sGamma, sBeta],
        [cAlpha * sGamma + cGamma * sAlpha * sBeta, cAlpha * cGamma - sAlpha * sBeta * sGamma, -cBeta * sAlpha],
        [sAlpha * sGamma - cAlpha * cGamma * sBeta, cGamma * sAlpha + cAlpha * sBeta * sGamma, cAlpha * cBeta]
    ])

    transformed = np.dot(point, rotation_matrix.T) + np.array([p[0], p[1], p[2]])

    return transformed

def gl_6dof(map_global, scancloud, groundtruth, algorithm, version_fitness, err_dis,unif_noise):
    """
    Global Localization Algorithm based on evolutonary metaheuristics

    Definimos los límites de búsqueda para cada grado de libertad, limites del
    mapa en traslación +- 6 grados para pitch y roll y 360º para yaw

    Get the axis-aligned bounding box
    """

    aabb = map_global.get_axis_aligned_bounding_box()

    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()

    x_min, y_min, z_min = min_bound
    x_max, y_max, z_max = max_bound

    mapmin=[x_min, y_min, z_min, -0.1, -0.1,-pi]
    mapmax=[x_max, y_max, z_max, 0.1, 0.1, pi]

    real_scan = add_noise_to_pc(scancloud, err_dis, unif_noise) # añadir ruido de sensor y de ambiente al scan


    print(Color.GREEN + f'\nPosicion real del robot[x, y, z, alpha, beta, theta]: ' +
        f'[{round(groundtruth[0], 4)}, {round(groundtruth[1], 4)}, {round(groundtruth[2], 4)}, ' +
        f'{round(groundtruth[3], 4)}, {round(groundtruth[4], 4)}, {round(groundtruth[5], 4)}]' + Color.END)
    
    initial_time = time.time()

    #--------------------------------------------------------------------------------------------
    # EJECUCION DEL ALGORITMO EVOLUTIVO

    if (algorithm.type == 1): # Differential Evolution
        NPini=algorithm.NPini
        iter_max=algorithm.iter_max
        D=algorithm.D
        F=algorithm.F
        CR=algorithm.CR
        [pcAligned, estimate, bestCost, rmse_array, it] = de_6dof(real_scan, map_global,mapmax,mapmin,err_dis,NPini,D,iter_max,F,CR,version_fitness)

    final_time = time.time()
    
    # Display results
    print(f'\nPosicion real del robot[x, y, z, alpha, beta, theta]: ' +
        f'[{round(groundtruth[0], 4)}, {round(groundtruth[1], 4)}, {round(groundtruth[2], 4)}, ' +
        f'{round(groundtruth[3], 4)}, {round(groundtruth[4], 4)}, {round(groundtruth[5], 4)}]')
    
    print(f'\nPosicion estimada tras ejecución: ' +
        f'[{round(estimate[0], 4)}, {round(estimate[1], 4)}, {round(estimate[2], 4)}, ' +
        f'{round(estimate[3], 4)}, {round(estimate[4], 4)}, {round(estimate[5], 4)}]')

    poserror = np.sqrt((groundtruth[0] - estimate[0]) ** 2 + (groundtruth[1] - estimate[1]) ** 2 + (groundtruth[2] - estimate[2]) ** 2)
    orierror = [
        abs((groundtruth[3] - estimate[3]) * 180 / pi),
        abs((groundtruth[4] - estimate[4]) * 180 / pi),
        abs((groundtruth[5] - estimate[5]) * 180 / pi)
    ]

    print(Color.PURPLE + f'\nEl error de posicion es: {round(poserror,4)} m y el de orientacion: [{round(orierror[0],4)}, {round(orierror[1],4)}, {round(orierror[2],4)}] grados' + Color.END)
    print(Color.GREEN + f'Tiempo transcurrido: {round(final_time-initial_time, 2)} segundos' + Color.END)
    solution = Solution(it, (final_time-initial_time), estimate, poserror, orierror, map_global, real_scan.points)

    return solution

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

def ask_cloud():
    num_clouds = len(os.listdir(LOCAL_CLOUDS_FOLDER))

    print(Color.BOLD + f'\nAvailable scans [1-{num_clouds}]' + Color.END)
    id_cloud = input(Color.BOLD + "Select cloud as real scan: " + Color.END)
    if not id_cloud.strip():
        id_cloud = 9
        print(f'Default selected cloud: {id_cloud}')
    try:
        if int(id_cloud) > num_clouds or int(id_cloud) < 1:
            print(f'Error. Selected cloud ({id_cloud}) does not exist.') 
            exit(1)

    except ValueError as e:
        print(f'Error: Invalid Number. {e}')
        exit(1)

    return id_cloud

def ask_params():
    # Simulated laser error
    err_dis = input(Color.BOLD + "\nSensor noise (%): " + Color.END)
    if not err_dis.strip():
        err_dis = 0
        print(f'Default Noise: {err_dis}%')
    else:
        try:
            err_dis = int(err_dis)
            
            if err_dis > 100 or err_dis < 0:
                print(f'Error. Selected error ({err_dis}) is invalid.') 
                exit(1)

            err_dis = err_dis/100

        except ValueError as e:
            print(f'Error: Invalid Input. {e}')
            exit(1)

    # Simulated environmental noise
    unif_noise = input(Color.BOLD + "\nEnvironmental noise (Uniform distribution) (%): " + Color.END)
    if not unif_noise.strip():
        unif_noise = 0
        print(f'Default Noise: 0%')
    else:
        try:
            unif_noise = int(unif_noise)
            
            if unif_noise > 100 or unif_noise < 0:
                print(f'Error. Selected error ({unif_noise}) is invalid.') 
                exit(1)

            unif_noise = unif_noise/100

        except ValueError as e:
            print(f'Error: Invalid Input. {e}')
            exit(1)

    # Algorithm selection
    algorithm_type = 1 # Differential Evolution (default)

    # Fitness Function Options:
    version_fitness = 1 # Sum of the squared errors (Default)

    ## ALGORITHM PARAMETERS SECTION ##
    print(Color.BOLD + f'\nDifferential Evolution parameters:\n' + Color.END)

    # Population size
    user_NPini = input(Color.BOLD + "Population size: " + Color.END)
    if not user_NPini.strip():
        user_NPini = 100
        print(f'Default population is {user_NPini}')
    else:
        try:
            user_NPini = int(user_NPini)
            
            if user_NPini <= 0:
                print(f'Error. Selected error ({user_NPini}) is invalid.') 
                exit(1)

        except ValueError as e:
            print(f'Error: Invalid Input. {e}')
            exit(1)

    # Max Iterations
    user_iter_max = input(Color.BOLD + "\nMax. iterations: " + Color.END)
    if not user_iter_max.strip():
        user_iter_max = 500
        print(f'Default iteration max is {user_iter_max}')
    else:
        try:
            user_iter_max = int(user_iter_max)
            
            if user_iter_max <= 0:
                print(f'Error. Selected error ({user_iter_max}) is invalid.') 
                exit(1)

        except ValueError as e:
            print(f'Error: Invalid Input. {e}')
            exit(1)

    return (err_dis, unif_noise, algorithm_type, version_fitness, user_NPini, user_iter_max)

def generate_point_cloud(auto=False,
                         id_cloud = 9,
                         err_dis = 0, 
                         unif_noise = 0,
                         algorithm_type = 1,
                         version_fitness = 1,
                         user_NPini = 100,
                         user_iter_max = 500,
                         D=6,
                         F=0.9,
                         CR=0.75):
    """
    Executes the evolutive localization algorithm.
    Returns the points that form the calculated point cloud.
    """

    map_global_ori = o3d.io.read_point_cloud(f"{PACKAGE_PATH}/map_global_ori.ply")

    # SELECT LOCAL POINTCLOUD
    real_scan_ori = o3d.io.read_point_cloud(f"{PACKAGE_PATH}/local_clouds/cloud_{id_cloud}.ply")

    ## READ GROUNDTRUTH ##

    groundtruth = get_groundtruth_data(GROUNDTRUTH_FILE_PATH, id_cloud)

    # Downsampling
    map_global = map_global_ori.uniform_down_sample(every_k_points=int(1 / DOWN_SAMPLING_FACTOR_GLOBAL)) # Original PointCloud (Global Map)
    real_scan = real_scan_ori.uniform_down_sample(every_k_points=int(1 / DOWN_SAMPLING_FACTOR))         # User Selected PointCloud (Local Map)

    # Variables introduced via keyboard # (Only if not in auto mode)
    if (auto):
        print("\n" + Color.DARKCYAN + f"Auto mode enabled. {id_cloud}/{len(os.listdir(LOCAL_CLOUDS_FOLDER))}" + Color.END)

    print(Color.BOLD + "\nFINAL ALGORITHM PARAMETERS: " + Color.END)
    print(f"Local Cloud: {id_cloud}")
    print(f"Sensor Error: {err_dis}")
    print(f"Uniform Noise: {unif_noise}")
    print(f"Algortihm type: {algorithm_type}")
    print(f"NPini: {user_NPini}")
    print(f"iter_max: {user_iter_max}")
    print(f"D: {D}")
    print(f"F: {F}")
    print(f"CR: {CR}")
    algorithm = Algorithm(type=algorithm_type, NPini=user_NPini, iter_max=user_iter_max, D=D, F=F, CR=CR)

    solution = gl_6dof(map_global, real_scan, groundtruth, algorithm, version_fitness, err_dis, unif_noise)

    #Plot Results
    sol_points = spatial_rotation(real_scan_ori.points, solution.pose_estimate)
    
    poserror = np.sqrt((groundtruth[0] - solution.pose_estimate[0]) ** 2 + (groundtruth[1] - solution.pose_estimate[1]) ** 2 + (groundtruth[2] - solution.pose_estimate[2]) ** 2)
    orierror = [
        abs((groundtruth[3] - solution.pose_estimate[3]) * 180 / pi),
        abs((groundtruth[4] - solution.pose_estimate[4]) * 180 / pi),
        abs((groundtruth[5] - solution.pose_estimate[5]) * 180 / pi)
    ]

    save_error_data(id_cloud, algorithm_type, user_NPini, user_iter_max, D, F, CR, solution.time, solution.it, poserror, orierror)

    return sol_points


def save_error_data(id_cloud, algorithm_type, user_NPini, user_iter_max, D, F, CR, time, it, poserror, orierror):
    """
    Saves the solution in the $HOME directory as a .csv file
    """

    home_route = os.path.expanduser("~")
    
    filename = "errordata.csv"
    filepath = os.path.join(home_route, filename)

    # Initializing CSV file with header if it doesn't exist
    if not os.path.exists(filepath):
        with open(filepath, mode='w', newline='') as archivo_csv:
            escritor_csv = csv.writer(archivo_csv)
            escritor_csv.writerow(['id_cloud', 'algorithm', 'NPini', 'iter_max', 'D', 'F', 'CR', 'time', 'it', 'poserror', 'orierror_1', 'orierror_2', 'orierror_2'])


    # Escribir los datos en el archivo CSV
    with open(filepath, mode='a', newline='') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)
        escritor_csv.writerow([id_cloud] + [algorithm_type] + [user_NPini] + [user_iter_max] + [D] + [F] + [CR] + [time] + [it] + [poserror] + orierror)
    
    print("\n"+Color.BOLD + "SUMMARY:" + Color.END)
    print(f"id_cloud: {id_cloud}")
    print(f"algorithm_type: {algorithm_type}")
    print(f"user_NPini: {user_NPini}")
    print(f"user_iter_max: {user_iter_max}")
    print(f"D: {D}")
    print(f"F: {F}")
    print(f"CR: {CR}")
    print(f"time: {time}s")
    print(f"it: {it}")
    print(f"poserror: {poserror}")
    print(f"orierror: {orierror}\n")
    print(Color.BOLD + f"Data Saved in {filepath}" + Color.END + "\n")

############################################################
############################################################
############################################################

class PCDPublisher(Node):

    def __init__(self):
        super().__init__('pcd_publisher_node')
        
        # Declara el parámetro mi_parametro con un valor predeterminado
        self.declare_parameter('auto', False)

        self.auto_mode = self.get_parameter('auto').value
        print(Color.CYAN + f"\nAuto Mode: {self.auto_mode}" + Color.END)

        self.pcd_publisher_local = self.create_publisher(sensor_msgs.PointCloud2, 'evloc_local', 10)
        self.pcd_publisher_global = self.create_publisher(sensor_msgs.PointCloud2, 'evloc_global', 10)

    def run(self):
        selcted_cloud = 1
        # Ask once before starting if in auto mode.
        if self.auto_mode:
            err_dis, unif_noise, algorithm_type, version_fitness, user_NPini, user_iter_max = ask_params()
        while True:

            print(Color.BOLD + "\n------------------------------------" + Color.END)

            # Ask every iteration if not in auto mode.
            if not self.auto_mode:
                selcted_cloud = ask_cloud()
                err_dis, unif_noise, algorithm_type, version_fitness, user_NPini, user_iter_max = ask_params()

            points = generate_point_cloud(auto=self.auto_mode,
                                          id_cloud = selcted_cloud,
                                          err_dis = err_dis, 
                                          unif_noise = unif_noise,
                                          algorithm_type = algorithm_type,
                                          version_fitness = version_fitness,
                                          user_NPini = user_NPini,
                                          user_iter_max = user_iter_max)

            if points is None:
                print("Error generating point cloud.")
                break
            
            self.publish_point_cloud(points, 'map')
            
            if not self.auto_mode:
                restart = self.ask_restart()
                if not restart:
                    self.destroy_node()  # Cierra el nodo antes de salir del bucle
                    break
            else:
                # Loop for every cloud when in auto mode
                selcted_cloud += 1
                if selcted_cloud > 44:
                    selcted_cloud = 1

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


    def publish_point_cloud(self, points, parent_frame):
        
        points = points[::2] # Downsampling. Son demasiados puntos para RVIZ
        pcd = self.point_cloud(points, parent_frame)
        self.pcd_publisher_local.publish(pcd)
        print(f"Local PointCloud with dimensions {points.shape} has been published.")

        map_global_ori = o3d.io.read_point_cloud(f"{PACKAGE_PATH}/map_global_ori.ply")
        points2 = np.asarray(map_global_ori.points)[::40] # Downsampling. Son demasiados puntos para RVIZ
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
    pcd_publisher = PCDPublisher()
    pcd_publisher.run()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
