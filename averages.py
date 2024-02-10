import pandas as pd
import os

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


# Obtener la ruta del directorio HOME
home_route = os.path.expanduser("~")

# Definir la ruta del archivo CSV
filename = "errordata.csv"
filepath = os.path.join(home_route, filename)

# Cargar el archivo CSV en un DataFrame de pandas
data = pd.read_csv(filepath)

# Ordenar el DataFrame por la primera columna (ID del objeto)
data_sorted = data.sort_values(by='id_cloud')

# Obtener los valores únicos de la primera columna
unique_ids = data_sorted['id_cloud'].unique()

# Iterar sobre los valores únicos de la primera columna
print(Color.YELLOW + "\nESTADÍSTICAS: \n" + Color.END)
for unique_id in unique_ids:
    # Filtrar el DataFrame por el valor único de la primera columna
    filtered_data = data_sorted[data_sorted['id_cloud'] == unique_id]
    
    # Calcular las estadísticas para este grupo de datos
    means = filtered_data.mean()
    medians = filtered_data.median()
    minimums = filtered_data.min()
    maximums = filtered_data.max()
    std_devs = filtered_data.std()
    
    # Imprimir las estadísticas para este grupo
    print("----------------------------------------------------")
    print(Color.GREEN + f"\nEstadísticas para la nube de puntos {unique_id}:\n" + Color.END)
    print(Color.BOLD + f"{'Columna':<12} {'Media':<12} {'Mediana':<12} {'Mínimo':<12} {'Máximo':<12} {'Desviación estándar':<20}" +Color.END)
    print("--------------------------------------------------------------------------------------")
    for column in filtered_data.columns[1:]:
        print(f"{Color.BOLD}{column:<12}{Color.END} {round(means[column], 4):<12} {round(medians[column], 4):<12} {round(minimums[column], 4):<12} {round(maximums[column], 4):<12} {round(std_devs[column], 4):<20}")
    
    # Imprimir el número de muestras en este grupo
    num_samples = len(filtered_data)
    print(Color.CYAN + f"\nNúmero de muestras en el grupo {unique_id}: {num_samples}" + Color.END)
    print("\n----------------------------------------------------")

# Imprimir el número total de filas en el DataFrame
print(Color.UNDERLINE + f"\nNúmero total de muestras: {data_sorted.shape[0]}\n" + Color.END)
