import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

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


#home_route = os.path.expanduser("~")

current_directory = os.path.dirname(__file__)
parent_directory = os.path.join(current_directory, '..')
filepath = os.path.join(parent_directory, 'errordata.csv')

MAX_POS_ERROR = 0.5 # In meters
MAX_ORI_ERROR = 5 # In degrees (mean of three ori errors)

########################################################################



def getSampleCounts(data_frame):
    sample_counts = data_frame.groupby('id_cloud').size()
    return sample_counts



def getPosErrorAvg(data_frame):
    ids = data_frame['id_cloud'].tolist()
    pos_errors = data_frame['poserror'].tolist()

    pares_ordenados = sorted(zip(ids, pos_errors))

    df_pares_ordenados = pd.DataFrame(pares_ordenados, columns=['id_cloud', 'poserror'])

    poserror_avg = df_pares_ordenados.groupby('id_cloud')['poserror'].mean()

    return poserror_avg



def getOriErrorAvg(data_frame):
    ids = data_frame['id_cloud'].tolist()
    orierror_1 = data_frame['orierror_1'].tolist()
    orierror_2 = data_frame['orierror_2'].tolist()
    orierror_3 = data_frame['orierror_3'].tolist()

    pares_ordenados_1 = sorted(zip(ids, orierror_1))
    pares_ordenados_2 = sorted(zip(ids, orierror_2))
    pares_ordenados_3 = sorted(zip(ids, orierror_3))

    df_pares_ordenados_1 = pd.DataFrame(pares_ordenados_1, columns=['id_cloud', 'orierror_1'])
    df_pares_ordenados_2 = pd.DataFrame(pares_ordenados_2, columns=['id_cloud', 'orierror_2'])
    df_pares_ordenados_3 = pd.DataFrame(pares_ordenados_3, columns=['id_cloud', 'orierror_3'])

    orierror_avg_1 = df_pares_ordenados_1.groupby('id_cloud')['orierror_1'].mean()
    orierror_avg_2 = df_pares_ordenados_2.groupby('id_cloud')['orierror_2'].mean()
    orierror_avg_3 = df_pares_ordenados_3.groupby('id_cloud')['orierror_3'].mean()

    total_orierror_avg = (orierror_avg_1 + orierror_avg_2 + orierror_avg_3) / 3

    return total_orierror_avg



def getItAvg(data_frame):
    ids = data_frame['id_cloud'].tolist()
    its = data_frame['it'].tolist()

    pares_ordenados = sorted(zip(ids, its))

    df_pares_ordenados = pd.DataFrame(pares_ordenados, columns=['id_cloud', 'it'])

    it_avg = df_pares_ordenados.groupby('id_cloud')['it'].mean()

    return it_avg



def getTimeAvg(data_frame):
    ids = data_frame['id_cloud'].tolist()
    times = data_frame['time'].tolist()

    pares_ordenados = sorted(zip(ids, times))

    df_pares_ordenados = pd.DataFrame(pares_ordenados, columns=['id_cloud', 'time'])

    time_avg = df_pares_ordenados.groupby('id_cloud')['time'].mean()

    return time_avg



def showBarCombined(dataframe, title, title_x, title_y, max_y_value=None,
                    custom_y_limit = None):
        # Graficar los datos con barras
        plt.figure(figsize=(14, 6))

        # Obtener los datos para graficar
        ids = dataframe.index
        bar_width = 0.2
        index = np.arange(len(ids))

        # Graficar cada algoritmo con un color diferente
        plt.bar(index, dataframe['DE'], bar_width, label='DE', color='blue')
        plt.bar(index + bar_width, dataframe['PSO'], bar_width, label='PSO', color='green')
        plt.bar(index + 2*bar_width, dataframe['IWO'], bar_width, label='IWO', color='red')

        # Configurar etiquetas y título
        plt.xlabel(title_x)
        plt.ylabel(title_y)
        plt.title(title)
        plt.xticks(index + bar_width, ids)
        plt.legend()

        if custom_y_limit:
            plt.axhline(y=custom_y_limit, color='gray', linestyle='--')

        if max_y_value:
            plt.ylim(0, max_y_value)

        # Mostrar la gráfica
        plt.tight_layout()
        plt.show()



def showBarSimple(dataframe, title, title_x, title_y, max_y_value=None, custom_y_limit=None, color_gradient=False):

    # Graficar los datos con barras
    plt.figure(figsize=(14, 6))

    # Obtener los datos para graficar
    ids = dataframe.index
    y_data = dataframe.values
    bar_width = 0.2
    index = np.arange(len(ids))

    if color_gradient:
        # Normalizar los valores Y
        norm = plt.Normalize(y_data.min(), y_data.max())
        cmap = plt.get_cmap('RdYlGn_r')  # Cambiado a 'RdYlGn'

        # Graficar las barras con colores gradiente
        for i, y in enumerate(y_data):
            color = cmap(norm(y))
            plt.bar(index[i], y, color=color)
    else:
        # Graficar las barras con un color sólido
        plt.bar(index, y_data)  # Puedes cambiar 'blue' al color que desees

    # Configurar etiquetas y título
    plt.xlabel(title_x)
    plt.ylabel(title_y)
    plt.title(title)
    plt.xticks(index + bar_width, ids)
    
    # Agregar línea discontinua si se especifica
    if custom_y_limit:
        plt.axhline(y=custom_y_limit, color='gray', linestyle='--')

    # Establecer límite en el eje y si se especifica
    if max_y_value:
        plt.ylim(0, max_y_value)

    # Mostrar la gráfica
    plt.tight_layout()
    plt.show()



def main():
    data = pd.read_csv(filepath)

    al_cont = data['algorithm'].value_counts()

    while True:
        user_selection = input(Color.BOLD + f"\nIntroduce the Evolutionary Algorithm that you want to analyze: \n\n"
                        f"1) DE - Differential Evolution (Default)  -  {al_cont.values[0]} samples\n"
                        f"2) PSO - Particle Swarm Optimization      -  {al_cont.values[1]} samples\n"
                        f"3) IWO - Invasive Weed Optimization       -  {al_cont.values[2]} samples\n"
                        f"4) ALL                                    -  {al_cont.values[0]+al_cont.values[1]+al_cont.values[2]} samples\n"
                        + Color.END)
        
        if user_selection in ['1', '2', '3', '4']:
            break
        else:
            print("Por favor, ingrese una opción válida (1, 2 o 3).")

    user_selection = int(user_selection)

    if user_selection == 1:
        print("Differential Evolution selected")
        de_data = data[data['algorithm'] == 1]

        num_samples_de = getSampleCounts(de_data)

        # Calcula las medias para cada algoritmo
        time_avg_de = getTimeAvg(de_data)

        it_avg_de = getItAvg(de_data)

        poserror_avg_de = getPosErrorAvg(de_data)

        orierror_avg_de = getOriErrorAvg(de_data)

        title_x = 'Cloud ID'

        title = 'Number of samples per cloud ID for DE algorithm'
        title_y = 'Number of samples'
        showBarSimple(num_samples_de, title, title_x, title_y, custom_y_limit=10)

        title = 'Execution time per cloud ID for DE algorithm'
        title_y = 'Execution time (seconds)'
        showBarSimple(time_avg_de, title, title_x, title_y, color_gradient=True)

        title = 'Iterations per cloud ID for DE algorithm'
        title_y = 'Iterations until convergence'
        showBarSimple(it_avg_de, title, title_x, title_y, color_gradient=True)

        title = 'Position error (m) per cloud ID for DE algorithm'
        title_y = 'Position error in meters'
        showBarSimple(poserror_avg_de, title, title_x, title_y, custom_y_limit=MAX_POS_ERROR, color_gradient=True)

        title = 'Orientation error per cloud ID for DE algorithm'
        title_y = 'Orientation error in meters'
        showBarSimple(orierror_avg_de, title, title_x, title_y, custom_y_limit=MAX_ORI_ERROR, color_gradient=True)

    elif user_selection == 2:
        print("Particle Swarm Optimization selected")
        pso_data = data[data['algorithm'] == 2]

        num_samples_pso = getSampleCounts(pso_data)

        # Calcula las medias para cada algoritmo
        time_avg_pso = getTimeAvg(pso_data)

        it_avg_pso = getItAvg(pso_data)

        poserror_avg_pso = getPosErrorAvg(pso_data)

        orierror_avg_pso = getOriErrorAvg(pso_data)

        title_x = 'Cloud ID'

        title = 'Number of samples per cloud ID for PSO algorithm'
        title_y = 'Number of samples'
        showBarSimple(num_samples_pso, title, title_x, title_y, custom_y_limit=10)

        title = 'Execution time per cloud ID for PSO algorithm'
        title_y = 'Execution time (seconds)'
        showBarSimple(time_avg_pso, title, title_x, title_y, color_gradient=True)

        title = 'Iterations per cloud ID for PSO algorithm'
        title_y = 'Iterations until convergence'
        showBarSimple(it_avg_pso, title, title_x, title_y, color_gradient=True)

        title = 'Position error (m) per cloud ID for PSO algorithm'
        title_y = 'Position error in meters'
        showBarSimple(poserror_avg_pso, title, title_x, title_y, custom_y_limit=MAX_POS_ERROR, color_gradient=True)

        title = 'Orientation error per cloud ID for PSO algorithm'
        title_y = 'Orientation error in meters'
        showBarSimple(orierror_avg_pso, title, title_x, title_y, custom_y_limit=MAX_ORI_ERROR, color_gradient=True)

    elif user_selection == 3:
        print("Invasive Weed Optimization selected")
        iwo_data = data[data['algorithm'] == 3]

        num_samples_iwo = getSampleCounts(iwo_data)

        # Calcula las medias para cada algoritmo
        time_avg_iwo = getTimeAvg(iwo_data)

        it_avg_iwo = getItAvg(iwo_data)

        poserror_avg_iwo = getPosErrorAvg(iwo_data)

        orierror_avg_iwo = getOriErrorAvg(iwo_data)

        title_x = 'Cloud ID'

        title = 'Number of samples per cloud ID for IWO algorithm'
        title_y = 'Number of samples'
        showBarSimple(num_samples_iwo, title, title_x, title_y, custom_y_limit=10)

        title = 'Execution time per cloud ID for IWO algorithm'
        title_y = 'Execution time (seconds)'
        showBarSimple(time_avg_iwo, title, title_x, title_y, color_gradient=True)

        title = 'Iterations per cloud ID for IWO algorithm'
        title_y = 'Iterations until convergence'
        showBarSimple(it_avg_iwo, title, title_x, title_y, color_gradient=True)

        title = 'Position error (m) per cloud ID for IWO algorithm'
        title_y = 'Position error in meters'
        showBarSimple(poserror_avg_iwo, title, title_x, title_y, custom_y_limit=MAX_POS_ERROR, color_gradient=True)

        title = 'Orientation error per cloud ID for IWO algorithm'
        title_y = 'Orientation error in meters'
        showBarSimple(orierror_avg_iwo, title, title_x, title_y, custom_y_limit=MAX_ORI_ERROR, color_gradient=True)

    elif user_selection == 4:
        print("All Selected")
        de_data = data[data['algorithm'] == 1]
        pso_data = data[data['algorithm'] == 2]
        iwo_data = data[data['algorithm'] == 3]

        num_samples_de = getSampleCounts(de_data)
        num_samples_pso = getSampleCounts(pso_data)
        num_samples_iwo = getSampleCounts(iwo_data)

        # Calcula las medias para cada algoritmo
        time_avg_de = getTimeAvg(de_data)
        time_avg_pso = getTimeAvg(pso_data)
        time_avg_iwo = getTimeAvg(iwo_data)

        it_avg_de = getItAvg(de_data)
        it_avg_pso = getItAvg(pso_data)
        it_avg_iwo = getItAvg(iwo_data)

        poserror_avg_de = getPosErrorAvg(de_data)
        poserror_avg_pso = getPosErrorAvg(pso_data)
        poserror_avg_iwo = getPosErrorAvg(iwo_data)

        orierror_avg_de = getOriErrorAvg(de_data)
        orierror_avg_pso = getOriErrorAvg(pso_data)
        orierror_avg_iwo = getOriErrorAvg(iwo_data)

        title_x = 'Cloud ID'

        title = 'Number of samples per cloud ID for each algorithm'
        title_y = 'Number of samples'
        number_of_samples_combined = pd.DataFrame({'DE': num_samples_de, 'PSO': num_samples_pso, 'IWO': num_samples_iwo})
        showBarCombined(number_of_samples_combined, title, title_x, title_y, custom_y_limit=10)

        title = 'Execution time per cloud ID for each algorithm'
        title_y = 'Execution time (seconds)'
        combined_time_avg = pd.DataFrame({'DE': time_avg_de, 'PSO': time_avg_pso, 'IWO': time_avg_iwo})
        showBarCombined(combined_time_avg, title, title_x, title_y)

        title = 'Iterations per cloud ID for each algorithm'
        title_y = 'Iterations until convergence'
        combined_it_avg = pd.DataFrame({'DE': it_avg_de, 'PSO': it_avg_pso, 'IWO': it_avg_iwo})
        showBarCombined(combined_it_avg, title, title_x, title_y)

        title = 'Position error (m) per cloud ID for each algorithm'
        title_y = 'Position error in meters'
        combined_poserror_avg = pd.DataFrame({'DE': poserror_avg_de, 'PSO': poserror_avg_pso, 'IWO': poserror_avg_iwo})
        showBarCombined(combined_poserror_avg, title, title_x, title_y, custom_y_limit=MAX_POS_ERROR)

        title = 'Orientation error per cloud ID for each algorithm'
        title_y = 'Orientation error in meters'
        combined_orierror_avg = pd.DataFrame({'DE': orierror_avg_de, 'PSO': orierror_avg_pso, 'IWO': orierror_avg_iwo})
        showBarCombined(combined_orierror_avg, title, title_x, title_y, custom_y_limit=MAX_ORI_ERROR)


if __name__ == '__main__':
    main()