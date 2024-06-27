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

# Naranja, azul, verde
all_colors = ['#E69F00', '#0072B2', '#009E73']

#home_route = os.path.expanduser("~")

current_directory = os.path.dirname(__file__)
parent_directory = os.path.join(current_directory, '..')
filepath = os.path.join(parent_directory, 'errordata.csv')

MAX_POS_ERROR = 0.25 # In meters
MAX_ORI_ERROR_1 = 8
MAX_ORI_ERROR_2 = 8
MAX_ORI_ERROR_3 = 8
MAX_ORI_ERROR_COMBINED = (MAX_ORI_ERROR_1 + MAX_ORI_ERROR_2 + MAX_ORI_ERROR_3) / 3 # In degrees (mean of three ori errors)
MIN_CONVERGENCE_PERCENTAGE = 80
########################################################################


def getSampleCounts(data_frame):
    sample_counts = data_frame.groupby('id_cloud').size()
    return sample_counts


def getConvPerc(data_frame):
    pos_errors = data_frame['poserror']
    ori_errors = data_frame[['orierror_1', 'orierror_2', 'orierror_3']]

    # Calcula la condición de convergencia
    converged = (pos_errors <= MAX_POS_ERROR) & (ori_errors <= [MAX_ORI_ERROR_1, MAX_ORI_ERROR_2, MAX_ORI_ERROR_3]).all(axis=1)

    total_samples = data_frame['id_cloud'].value_counts()

    convergence_counts = converged.groupby(data_frame['id_cloud']).sum()

    convergence_percentage = (convergence_counts / total_samples) * 100

    return convergence_percentage


def getOriErrorStd(data_frame):
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

    orierror_std_1 = df_pares_ordenados_1.groupby('id_cloud')['orierror_1'].std()
    orierror_std_2 = df_pares_ordenados_2.groupby('id_cloud')['orierror_2'].std()
    orierror_std_3 = df_pares_ordenados_3.groupby('id_cloud')['orierror_3'].std()

    total_orierror_std = (orierror_std_1 + orierror_std_2 + orierror_std_3) / 3

    return total_orierror_std


def getOriErrorStats(data_frame):
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

    orierror_stats_1 = df_pares_ordenados_1.groupby('id_cloud')['orierror_1'].median()
    orierror_stats_2 = df_pares_ordenados_2.groupby('id_cloud')['orierror_2'].median()
    orierror_stats_3 = df_pares_ordenados_3.groupby('id_cloud')['orierror_3'].median()

    total_orierror_stats = (orierror_stats_1 + orierror_stats_2 + orierror_stats_3) / 3

    return total_orierror_stats


def getDataStats(data_frame, col_id):

    ids = data_frame['id_cloud'].tolist()
    column_data = data_frame[col_id].tolist()

    pares_ordenados = sorted(zip(ids, column_data))

    df_pares_ordenados = pd.DataFrame(pares_ordenados, columns=['id_cloud', col_id])

    data_stats = df_pares_ordenados.groupby('id_cloud')[col_id].median()

    return data_stats



def showBarCombined(dataframe, title, title_x, title_y, max_y_value=None,
                    custom_y_limit = None):
    mid_index = len(dataframe) // 2
    dataframe_part1 = dataframe.iloc[:22]
    dataframe_part2 = dataframe.iloc[22:44]

    # Función interna para graficar una parte del dataframe
    def plot_part(dataframe_part, title_suffix):
        plt.figure(figsize=(14, 6))
        
        # Obtener los datos para graficar
        ids = dataframe_part.index
        bar_width = 0.2
        index = np.arange(len(ids))

        # Convertir los datos a arrays de numpy
        de_values = dataframe_part['DE'].values
        pso_values = dataframe_part['PSO'].values
        iwo_values = dataframe_part['IWO'].values

        # Crear las barras agrupadas
        plt.bar(index, de_values, bar_width, label='DE', color=all_colors[0])
        plt.bar(index + bar_width, pso_values, bar_width, label='PSO', color=all_colors[1])
        plt.bar(index + 2 * bar_width, iwo_values, bar_width, label='IWO', color=all_colors[2])

        if custom_y_limit:
            plt.axhline(y=custom_y_limit, color='gray', linestyle='--')

        # Configurar etiquetas y título
        plt.xlabel(title_x)
        plt.ylabel(title_y)
        plt.title(f"{title} - {title_suffix}")
        plt.xticks(index + bar_width, ids)
        plt.legend()

        # Mostrar la gráfica
        plt.tight_layout()
        plt.show()

    # Graficar la primera parte
    plot_part(dataframe_part1, "Primeras 22 Medidas")

    # Graficar la segunda parte
    plot_part(dataframe_part2, "Últimas 22 Medidas")



def showErrorBarSimple(dataframe_avg, stds, title, title_x, title_y, max_y_value=None, custom_y_limit=None, color_gradient=False, color_threshold=False):

    # Graficar los datos con error bars
    plt.figure(figsize=(14, 6))

    # Obtener los datos para graficar
    ids = dataframe_avg.index
    y_data = dataframe_avg.values
    x_positions = np.arange(len(ids))

    # Trazar la gráfica con error bars
    plt.errorbar(ids, y_data, yerr=stds, fmt='o', capsize=5)

    # Configurar etiquetas y título
    plt.xlabel(title_x)
    plt.ylabel(title_y)
    plt.title(title)
    plt.xticks(x_positions + 1, ids)

    # Agregar línea discontinua si se especifica
    if custom_y_limit:
        plt.axhline(y=custom_y_limit, color='gray', linestyle='--')

    # Establecer límite en el eje y si se especifica
    if max_y_value:
        plt.ylim(0, max_y_value)

    # Mostrar la gráfica
    plt.tight_layout()
    plt.show()


def getDataStd(data_frame, col_id):
    ids = data_frame['id_cloud'].tolist()
    column_data = data_frame[col_id].tolist()

    pares_ordenados = sorted(zip(ids, column_data))
    df_pares_ordenados = pd.DataFrame(pares_ordenados, columns=['id_cloud', col_id])

    # Calcular el rango intercuartil (IQR)
    Q1 = df_pares_ordenados.groupby('id_cloud')[col_id].quantile(0.25)
    Q3 = df_pares_ordenados.groupby('id_cloud')[col_id].quantile(0.75)
    IQR = Q3 - Q1

    # Calcular el rango medio absoluto de la desviación (MAD)
    MAD = df_pares_ordenados.groupby('id_cloud')[col_id].apply(lambda x: np.median(np.abs(x - x.median())))

    # Calcular la desviación estándar
    STD = df_pares_ordenados.groupby('id_cloud')[col_id].std()

    return STD


def showBarSimple(dataframe, title, title_x, title_y, max_y_value=None, custom_y_limit=None, color_gradient=False, color_threshold=False):

    # Graficar los datos con barras
    plt.figure(figsize=(14, 6))

    # Obtener los datos para graficar
    ids = dataframe.index
    y_data = dataframe.values
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
        if color_threshold:
            colors = [all_colors[1] if y < MIN_CONVERGENCE_PERCENTAGE else all_colors[0] for y in y_data]
            plt.bar(index, y_data, color=colors)
        else:
            # Graficar las barras con un color sólido
            plt.bar(index, y_data)  # Puedes cambiar 'blue' al color que desees

    # Configurar etiquetas y título
    plt.xlabel(title_x)
    plt.ylabel(title_y)
    plt.title(title)
    plt.xticks(index, ids)
    
    # Agregar línea discontinua si se especifica
    if custom_y_limit:
        plt.axhline(y=custom_y_limit, color='gray', linestyle='--')

    # Establecer límite en el eje y si se especifica
    if max_y_value:
        plt.ylim(0, max_y_value)

    # Mostrar la gráfica
    plt.tight_layout()
    plt.show()



def showInfo(data_frame, name):
    num_samples_df = getSampleCounts(data_frame)

    # Calcula las medias para cada algoritmo
    time_avg_df = getDataStats(data_frame, 'time')
    time_std_df = getDataStd(data_frame, 'time')

    it_avg_df = getDataStats(data_frame, 'it')
    it_std_df = getDataStd(data_frame, 'it')

    poserror_avg_df = getDataStats(data_frame, 'poserror')
    poserror_std_df = getDataStd(data_frame, 'poserror')

    orierror_stats_df = getOriErrorStats(data_frame)
    orierror_std_df = getOriErrorStd(data_frame)

    conv_perc_df = getConvPerc(data_frame)
    num_convergences = (conv_perc_df >= MIN_CONVERGENCE_PERCENTAGE).sum()
    print(Color.CYAN + f"{name} Convergences: {num_convergences}/{conv_perc_df.count()}" + Color.END)

    title_x = 'Nube de puntos'

    title = f'Porcentaje de convergencia por nube de puntos. Algortimo {name}. ({MAX_POS_ERROR}m, {MAX_ORI_ERROR_1}º, {MAX_ORI_ERROR_2}º, {MAX_ORI_ERROR_3}º)'
    title_y = 'Porcentaje de convergencia'
    showBarSimple(conv_perc_df, title, title_x, title_y, custom_y_limit=MIN_CONVERGENCE_PERCENTAGE, color_threshold=True)

    title = f'Number of samples per cloud ID for {name} algorithm'
    title_y = 'Number of samples'
    showBarSimple(num_samples_df, title, title_x, title_y, custom_y_limit=10)

    title = f'Tiempo de ejecución por nube de puntos. Algoritmo {name}'
    title_y = 'Tiempo de ejecución (segundos)'
    showErrorBarSimple(time_avg_df, time_std_df, title, title_x, title_y, color_gradient=True)

    title = f'Iteraciones por nube de puntos. Algoritmo {name}'
    title_y = 'Iteraciones'
    showErrorBarSimple(it_avg_df, it_std_df, title, title_x, title_y)

    title = f'Error de posición por nube de puntos. Algoritmo {name}'
    title_y = 'Error de posición (metros)'
    showErrorBarSimple(poserror_avg_df, poserror_std_df, title, title_x, title_y, custom_y_limit=MAX_POS_ERROR, color_gradient=True)

    title = f'Error de orientación por nube de puntos. Algoritmo {name}'
    title_y = 'Error de orientación (grados)'
    showErrorBarSimple(orierror_stats_df, orierror_std_df, title, title_x, title_y, custom_y_limit=MAX_ORI_ERROR_COMBINED, color_gradient=True)


def main():
    data = pd.read_csv(filepath)

    al_cont = data['algorithm'].value_counts()

    while True:
        user_selection = input(Color.BOLD + f"\nIntroduce the Evolutionary Algorithm that you want to analyze: \n\n"
                        f"1) DE   -  Differential Evolution       -  {al_cont.values[0]} samples\n"
                        f"2) PSO  -  Particle Swarm Optimization  -  {al_cont.values[1]} samples\n"
                        f"3) IWO  -  Invasive Weed Optimization   -  {al_cont.values[2]} samples\n"
                        f"4) ALL  -  Analyze all algorithms       -  {al_cont.values[0]+al_cont.values[1]+al_cont.values[2]} samples\n"
                        + Color.END)
        
        if user_selection in ['1', '2', '3', '4']:
            break
        else:
            print("Por favor, ingrese una opción válida (1, 2 o 3).")

    user_selection = int(user_selection)

    if user_selection == 1:
        print("Differential Evolution selected")
        de_data = data[data['algorithm'] == 1]
        showInfo(de_data, 'DE')

    elif user_selection == 2:
        print("Particle Swarm Optimization selected")
        pso_data = data[data['algorithm'] == 2]
        showInfo(pso_data, 'PSO')

    elif user_selection == 3:
        print("Invasive Weed Optimization selected")
        iwo_data = data[data['algorithm'] == 3]
        showInfo(iwo_data, 'IWO')

    elif user_selection == 4:
        print("All Selected")
        de_data = data[data['algorithm'] == 1]
        pso_data = data[data['algorithm'] == 2]
        iwo_data = data[data['algorithm'] == 3]

        conv_perc_de = getConvPerc(de_data)
        conv_perc_pso = getConvPerc(pso_data)
        conv_perc_iwo = getConvPerc(iwo_data)

        num_samples_de = getSampleCounts(de_data)
        num_samples_pso = getSampleCounts(pso_data)
        num_samples_iwo = getSampleCounts(iwo_data)

        # Calcula las medias para cada algoritmo
        time_avg_de = getDataStats(de_data, 'time')
        time_avg_pso = getDataStats(pso_data, 'time')
        time_avg_iwo = getDataStats(iwo_data, 'time')

        print(f"DE Time avg: {time_avg_de.mean()}")
        print(f"PSO Time avg: {time_avg_pso.mean()}")
        print(f"IWO Time avg: {time_avg_iwo.mean()}")

        it_avg_de = getDataStats(de_data, 'it')
        it_avg_pso = getDataStats(pso_data, 'it')
        it_avg_iwo = getDataStats(iwo_data, 'it')

        print(f"DE IT avg: {it_avg_de.mean()}")
        print(f"PSO IT avg: {it_avg_pso.mean()}")
        print(f"IWO IT avg: {it_avg_iwo.mean()}")

        poserror_avg_de = getDataStats(de_data, 'poserror')
        poserror_avg_pso = getDataStats(pso_data, 'poserror')
        poserror_avg_iwo = getDataStats(iwo_data, 'poserror')

        orierror1_stats_de = getDataStats(de_data, 'orierror_1')
        orierror1_stats_pso = getDataStats(pso_data, 'orierror_1')
        orierror1_stats_iwo = getDataStats(iwo_data, 'orierror_1')

        orierror2_stats_de = getDataStats(de_data, 'orierror_2')
        orierror2_stats_pso = getDataStats(pso_data, 'orierror_2')
        orierror2_stats_iwo = getDataStats(iwo_data, 'orierror_2')

        orierror3_stats_de = getDataStats(de_data, 'orierror_3')
        orierror3_stats_pso = getDataStats(pso_data, 'orierror_3')
        orierror3_stats_iwo = getDataStats(iwo_data, 'orierror_3')

        num_convergences_de = (conv_perc_de >= MIN_CONVERGENCE_PERCENTAGE).sum()
        num_convergences_pso = (conv_perc_pso >= MIN_CONVERGENCE_PERCENTAGE).sum()
        num_convergences_iwo = (conv_perc_iwo >= MIN_CONVERGENCE_PERCENTAGE).sum()
        print(Color.CYAN + f"DE Convergences: {num_convergences_de}/{conv_perc_de.count()}" + Color.END)
        print(Color.CYAN + f"PSO Convergences: {num_convergences_pso}/{conv_perc_pso.count()}" + Color.END)
        print(Color.CYAN + f"IWO Convergences: {num_convergences_iwo}/{conv_perc_iwo.count()}" + Color.END)

        title_x = 'Nube de puntos'

        title = f'Porcentaje de convergencia por nube de puntos por algortimo. ({MAX_POS_ERROR}m, {MAX_ORI_ERROR_1}º, {MAX_ORI_ERROR_2}º, {MAX_ORI_ERROR_3}º)'
        title_y = 'Porcentaje de convergencia'
        combined_conv_perc = pd.DataFrame({'DE': conv_perc_de, 'PSO': conv_perc_pso, 'IWO': conv_perc_iwo})
        showBarCombined(combined_conv_perc, title, title_x, title_y, custom_y_limit=MIN_CONVERGENCE_PERCENTAGE)

        # title = 'Number of samples per cloud ID for each algorithm'
        # title_y = 'Number of samples'
        # number_of_samples_combined = pd.DataFrame({'DE': num_samples_de, 'PSO': num_samples_pso, 'IWO': num_samples_iwo})
        # showBarCombined(number_of_samples_combined, title, title_x, title_y, custom_y_limit=10)

        title = 'Tiempo de ejecución por nube de puntos por algortimo'
        title_y = 'Tiempo de ejecución (segundos)'
        combined_time_avg = pd.DataFrame({'DE': time_avg_de, 'PSO': time_avg_pso, 'IWO': time_avg_iwo})
        showBarCombined(combined_time_avg, title, title_x, title_y)

        title = 'Iteraciones por nube de puntos por algortimo'
        title_y = 'Iteraciones'
        combined_it_avg = pd.DataFrame({'DE': it_avg_de, 'PSO': it_avg_pso, 'IWO': it_avg_iwo})
        showBarCombined(combined_it_avg, title, title_x, title_y)

        title = 'Error de posición por nube de puntos por algortimo'
        title_y = 'Error de posición en metros'
        combined_poserror_avg = pd.DataFrame({'DE': poserror_avg_de, 'PSO': poserror_avg_pso, 'IWO': poserror_avg_iwo})
        showBarCombined(combined_poserror_avg, title, title_x, title_y, custom_y_limit=MAX_POS_ERROR)

        title = 'Error de orientación (Cabeceo) por nube de puntos por algortimo'
        title_y = 'Error de orientación en grados'
        combined_orierror_stats = pd.DataFrame({'DE': orierror1_stats_de, 'PSO': orierror1_stats_pso, 'IWO': orierror1_stats_iwo})
        showBarCombined(combined_orierror_stats, title, title_x, title_y, custom_y_limit=MAX_ORI_ERROR_COMBINED)

        title = 'Error de orientación (Alabeo) por nube de puntos por algortimo'
        title_y = 'Error de orientación en grados'
        combined_orierror_stats = pd.DataFrame({'DE': orierror2_stats_de, 'PSO': orierror2_stats_pso, 'IWO': orierror2_stats_iwo})
        showBarCombined(combined_orierror_stats, title, title_x, title_y, custom_y_limit=MAX_ORI_ERROR_COMBINED)

        title = 'Error de orientación (Guiñada) por nube de puntos por algortimo'
        title_y = 'Error de orientación en grados'
        combined_orierror_stats = pd.DataFrame({'DE': orierror3_stats_de, 'PSO': orierror3_stats_pso, 'IWO': orierror3_stats_iwo})
        showBarCombined(combined_orierror_stats, title, title_x, title_y, custom_y_limit=MAX_ORI_ERROR_COMBINED)

if __name__ == '__main__':
    main()
