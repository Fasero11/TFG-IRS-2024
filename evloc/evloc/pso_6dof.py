import numpy as np
import open3d as o3d
import time
from evloc.common_functions import costfunction3d
from evloc.common_functions import distance_pc_to_point
from evloc.common_functions import spatial_rotation
from evloc.common_classes import Color

class Particle:
    """
    Part of the PSO algortihm
    """
    def __init__(self):
        self.Position = np.array([])
        self.Cost = None
        self.Count = None
        self.Velocity = np.array([])
        self.Best = BestParticle_()

class BestParticle_:
    """
    Part of the PSO algortihm
    """
    def __init__(self):
        self.Position = np.array([])
        self.Cost = None

def pso_6dof(scanCloud,mapCloud,mapmax,mapmin,err_dis,NPini,D, w, wdamp, c1, c2, iter_max, version_fitness):
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

    # PSO Parameters
    nPop = NPini
    minIt = 50  # Minimum number of iterations

    # Velocity Limits. Definen cuanto se puede desplazar cada particula entre
    # iteraciones (en este caso un porcentaje 10% de la medida de cada dimension)

    VelMax_x = 0.1 * (higherBoundX - lowerBoundX)
    VelMin_x = -VelMax_x

    VelMax_y = 0.1 * (higherBoundY - lowerBoundY)
    VelMin_y = -VelMax_y

    VelMax_z = 0.1 * (higherBoundZ - lowerBoundZ)
    VelMin_z = -VelMax_z

    VelMax_Rx = 0.1 * (higherAngle_rx - lowerAngle_rx)
    VelMin_Rx = -VelMax_Rx

    VelMax_Ry = 0.1 * (higherAngle_ry - lowerAngle_ry)
    VelMin_Ry = -VelMax_Ry

    VelMax_Rz = 0.1 * (higherAngle_rz - lowerAngle_rz)
    VelMin_Rz = -VelMax_Rz

    # Initialization

    # Initialize Population
    particle = [Particle() for _ in range(nPop)]  # Crear nPop instancias de Particle
    GlobalBest = BestParticle_()
    GlobalBest.Cost = float('inf')  # Inicializamos el coste a infinito
    GlobalBest.Count = float('inf')  # Inicializamos el contador a infinito

    rndParticle = np.zeros(6)
    count = 0

    bestParticleCost = 100000000
    worstParticleCost = 100000
    count_bestfix = 0  # contadores a 0 para la convergencia del algoritmo
    count_worsefix = 0
    count_avgfix = 0
    ind_reparto_error = 100000

    # Initialize Position
    for current_iteration in range(nPop):
        if current_iteration == 0:  # first population is zero
            particle[current_iteration].Position = np.zeros(6)
        else:
            for n in range(nVar):
                if n == 0:  # Translation
                    rndParticle[n] = np.random.uniform(lowerBoundX, higherBoundX)
                elif n == 1:
                    rndParticle[n] = np.random.uniform(lowerBoundY, higherBoundY)
                elif n == 2:
                    rndParticle[n] = np.random.uniform(lowerBoundZ, higherBoundZ)
                elif n == 3:  # Angle
                    rndParticle[n] = np.random.uniform(lowerAngle_rx, higherAngle_rx)
                elif n == 4:
                    rndParticle[n] = np.random.uniform(lowerAngle_ry, higherAngle_ry)
                elif n == 5:
                    rndParticle[n] = np.random.uniform(lowerAngle_rz, higherAngle_rz)

            particle[current_iteration].Position = rndParticle.copy()

        # Initialize Velocity to zero
        particle[current_iteration].Velocity = np.zeros(VarSize)[0]

        # Evaluation poblacion inicial
        cand_scan = o3d.geometry.PointCloud()
        cand_scan.points = o3d.utility.Vector3dVector(spatial_rotation(scanCloud.points, particle[current_iteration].Position))

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
        dist_NNmap = distance_pc_to_point(correspondence_mat, particle[current_iteration].Position)
        dist_scancand = distance_pc_to_point(cand_scan.points, particle[current_iteration].Position)

        # Evaluar y asignar el error de las medidas (distancia euclídea o absoluta)
        particle[current_iteration].Cost = costfunction3d(dist_scancand, dist_NNmap, version_fitness, err_dis)

        # Update Personal Best
        particle[current_iteration].Best.Position = particle[current_iteration].Position
        particle[current_iteration].Best.Cost = particle[current_iteration].Cost

        # Update Global Best
        if particle[current_iteration].Best.Cost < GlobalBest.Cost:
            GlobalBest = particle[current_iteration].Best

        # Matriz para almacenar el mejor coste de cada iteración
        BestCosts = np.zeros(iter_max)


    #.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#
    # Bucle principal del algoritmo PSO #
    #.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#
        
    for it in range(iter_max):
        start_time = time.time()
        for pop_id in range(nPop):

            # La nueva velocidad depende de su velocidad anterior, de su distancia a la mejor posición histórica de todo el enjambre y a la mejor
            # posición histórica de ella misma. Los coeficientes w, c1 y c2 son parámetros que cuantifican la importancia
            # que se le da a cada parte. np.random.rand() genera un valor entre 0-1 (para cada parámetro) que introduce una componente aleatoria.

            particle[pop_id].Velocity = w * particle[pop_id].Velocity \
                + c1 * np.random.rand(*VarSize) * (particle[pop_id].Best.Position - particle[pop_id].Position) \
                + c2 * np.random.rand(*VarSize) * (GlobalBest.Position - particle[pop_id].Position)


            particle[pop_id].Velocity = particle[pop_id].Velocity[0]

            # Apply Velocity Limits

            particle[pop_id].Velocity[0] = max(particle[pop_id].Velocity[0], VelMin_x)
            particle[pop_id].Velocity[0] = min(particle[pop_id].Velocity[0], VelMax_x)

            particle[pop_id].Velocity[1] = max(particle[pop_id].Velocity[1], VelMin_y)
            particle[pop_id].Velocity[1] = min(particle[pop_id].Velocity[1], VelMax_y)

            particle[pop_id].Velocity[2] = max(particle[pop_id].Velocity[2], VelMin_z)
            particle[pop_id].Velocity[2] = min(particle[pop_id].Velocity[2], VelMax_z)

            particle[pop_id].Velocity[3] = max(particle[pop_id].Velocity[3], VelMin_Rx)
            particle[pop_id].Velocity[3] = min(particle[pop_id].Velocity[3], VelMax_Rx)

            particle[pop_id].Velocity[4] = max(particle[pop_id].Velocity[4], VelMin_Ry)
            particle[pop_id].Velocity[4] = min(particle[pop_id].Velocity[4], VelMax_Ry)

            particle[pop_id].Velocity[5] = max(particle[pop_id].Velocity[5], VelMin_Rz)
            particle[pop_id].Velocity[5] = min(particle[pop_id].Velocity[5], VelMax_Rz)

            # Update Position
            particle[pop_id].Position = np.array(particle[pop_id].Position) + np.array(particle[pop_id].Velocity)

            # Velocity Mirror Effect
            IsOutside = (
                (particle[pop_id].Position[0] < lowerBoundX) | (particle[pop_id].Position[0] > higherBoundX) |
                (particle[pop_id].Position[1] < lowerBoundY) | (particle[pop_id].Position[1] > higherBoundY) |
                (particle[pop_id].Position[2] < lowerBoundZ) | (particle[pop_id].Position[2] > higherBoundZ) |
                (particle[pop_id].Position[3] < lowerAngle_rx) | (particle[pop_id].Position[3] > higherAngle_rx) |
                (particle[pop_id].Position[4] < lowerAngle_ry) | (particle[pop_id].Position[4] > higherAngle_ry) |
                (particle[pop_id].Position[5] < lowerAngle_rz) | (particle[pop_id].Position[5] > higherAngle_rz)
            )

            # Update Velocity based on IsOutside
            if IsOutside:
                for j in range(6):
                    particle[pop_id].Velocity[j] = -particle[pop_id].Velocity[j]


            # Apply Position Limits
            particle[pop_id].Position[0] = max(min(particle[pop_id].Position[0], higherBoundX), lowerBoundX)
            particle[pop_id].Position[1] = max(min(particle[pop_id].Position[1], higherBoundY), lowerBoundY)
            particle[pop_id].Position[2] = max(min(particle[pop_id].Position[2], higherBoundZ), lowerBoundZ)
            particle[pop_id].Position[3] = max(min(particle[pop_id].Position[3], higherAngle_rx), lowerAngle_rx)
            particle[pop_id].Position[4] = max(min(particle[pop_id].Position[4], higherAngle_ry), lowerAngle_ry)
            particle[pop_id].Position[5] = max(min(particle[pop_id].Position[5], higherAngle_rz), lowerAngle_rz)


            # Evaluación nuevamente
            cand_scan = o3d.geometry.PointCloud()
            cand_scan.points = o3d.utility.Vector3dVector(spatial_rotation(scanCloud.points, particle[pop_id].Position))

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
            dist_NNmap = distance_pc_to_point(correspondence_mat, particle[pop_id].Position)
            dist_scancand = distance_pc_to_point(cand_scan.points, particle[pop_id].Position)

            #print(f"dist_NNmap: {dist_NNmap}  |  dist_scancand: {dist_scancand}")

            particle[pop_id].Cost = costfunction3d(dist_scancand, dist_NNmap, version_fitness, err_dis)

            # Update Personal Best
            if particle[pop_id].Cost < particle[pop_id].Best.Cost:
                particle[pop_id].Best.Position = particle[pop_id].Position
                particle[pop_id].Best.Cost = particle[pop_id].Cost

            # Update Global Best
            if particle[pop_id].Best.Cost < GlobalBest.Cost:
                GlobalBest = particle[pop_id].Best

            #print(f"It: {it} | particle[{pop_id}].Cost: " + str(particle[pop_id].Cost) + " GlobalBest.Cost: " + str(GlobalBest.Cost))


        BestCosts[it] = GlobalBest.Cost
        # Reducir la inercia
        w *= wdamp

        # Analizar la población
        sumcosts = 0  # Costo promedio
        for j in range(nPop):
            sumcosts += particle[j].Cost

        id = 0
        worst = 0
        worst_id = 0
        for p in particle:
            if p.Cost > worst:
                worst = p.Cost
                worst_id = id
            id += 1

        bestParticleCostnow = min(p.Cost for p in particle)
        worstParticleCostnow = max(p.Cost for p in particle)

        # Display evolution each 10 iterations
        if count == 10:
            print(f"\nIt: {it}, {Color.GREEN}Best: {round(GlobalBest.Cost, 4)}{Color.END}, {Color.RED}Worse: {round(worstParticleCost,4)}{Color.END}, {Color.YELLOW}Average: {round(sumcosts/nPop,4)}{Color.END}, Best/measure: {round(bestParticleCost/nPop,4)}, Worse/best: {round(worstParticleCost/bestParticleCost,4)}, Avg/best: {round(sumcosts/nPop/bestParticleCost,4)} \n Position (x, y, z, alpha, beta, theta): [{round(GlobalBest.Position[0],4)}, {round(GlobalBest.Position[1],4)}, {round(GlobalBest.Position[2],4)}, {round(GlobalBest.Position[3],4)}, {round(GlobalBest.Position[4],4)}, {round(GlobalBest.Position[5],4)}]\n")
            count=0
        count=count+1
        end_time = time.time()

        # Convergence Indicators
        if bestParticleCostnow < bestParticleCost:  # ¿Mejora el mejor respecto a la iteración anterior?
            count_worsefix = 0
            count_avgfix = 0
            count_bestfix = 0  # Sí, reiniciar contador a 0
        else:
            count_bestfix += 1  # No, incrementar contador de veces que no mejora

        bestParticleCost = bestParticleCostnow


        if worstParticleCost > worstParticleCostnow:  # ¿Mejora el peor candidato?
            count_worsefix = 0
            count_avgfix = 0
            count_bestfix = 0  # Sí, reiniciar contador a 0
        else:
            count_worsefix += 1  # No, incrementar contador de veces que no mejora

        worstParticleCost = worstParticleCostnow

        ind_reparto_error_aux = sumcosts / (nPop * bestParticleCost)

        if ind_reparto_error_aux < ind_reparto_error:  # ¿Mejora la media?
            count_avgfix = 0
            count_worsefix = 0
            count_bestfix = 0  # Sí, reiniciar contadores
        else:
            count_avgfix += 1  # No, incrementar contador de veces que no mejora

        ind_reparto_error = ind_reparto_error_aux

        #print(f"It: {it}, Time: {round(end_time-start_time,2)}  |||  count_bestfix: {count_bestfix}, count_worsefix: {count_worsefix}, count_avgfix: {count_avgfix}  |||  worstParticleCost: {worstParticleCost} | bestParticleCost: {bestParticleCost} | ind_reparto_error: {ind_reparto_error}")

        if (all([p.Cost for p in particle]) == GlobalBest.Cost) or \
        ((count_bestfix > 10 and count_worsefix > 10 and count_avgfix > 10) and it >= minIt) or \
        ((worstParticleCost / bestParticleCost < 1.15 and ind_reparto_error < 1.15) and it >= minIt):

            if all([p.Cost for p in particle]) == GlobalBest.Cost:
                stringcondition = 'total convergence'
            elif worstParticleCost / bestParticleCost < 1.15 and ind_reparto_error < 1.15:
                stringcondition = 'normal convergence'
            elif count_bestfix > 10 and count_worsefix > 10 and count_avgfix > 10:
                stringcondition = 'invariant convergence'

            print(f'Population converged in: {it} iterations and condition: {stringcondition}')
            break

    

    ########################################################
    ########################################################
    
    BestSol = GlobalBest

    BestParticle = BestSol.Position
    bestCost = BestSol.Cost
    rmse_array =  BestCosts

    pcAligned = o3d.geometry.PointCloud()
    pcAligned.points = o3d.utility.Vector3dVector(spatial_rotation(scanCloud.points, BestParticle))

    return(pcAligned, BestParticle, bestCost, rmse_array, it, stringcondition)