import pandas as pd
import os

# Límites de convergencia
POSERROR_THRESHOLD = 0.5
ORIERRROR_1_THRESHOLD = 5.0
ORIERRROR_2_THRESHOLD = 5.0
ORIERRROR_3_THRESHOLD = 5.0

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

home_route = os.path.expanduser("~")

filename = "errordata.csv"
filepath = os.path.join(home_route, filename)

data = pd.read_csv(filepath)

# Ordenar el DataFrame por la primera columna (ID del objeto)
data_sorted = data.sort_values(by='id_cloud')

# Obtener los valores únicos de la primera columna
unique_ids = data_sorted['id_cloud'].unique()

# Crear un diccionario para almacenar el recuento de errores y el número total de muestras por grupo
error_counts = {}

# Iterar sobre los valores únicos de la primera columna
for unique_id in unique_ids:
    # Filtrar el DataFrame por el valor único de la primera columna
    filtered_data = data_sorted[data_sorted['id_cloud'] == unique_id]
    
    # Contar el número de veces que todas las columnas son menores que sus umbrales correspondientes
    error_count = ((filtered_data['poserror'] < POSERROR_THRESHOLD) &
                   (filtered_data['orierror_1'] < ORIERRROR_1_THRESHOLD) &
                   (filtered_data['orierror_2'] < ORIERRROR_2_THRESHOLD) &
                   (filtered_data['orierror_3'] < ORIERRROR_3_THRESHOLD)).sum()
    
    # Obtener el número total de muestras en este grupo
    total_samples = len(filtered_data)
    
    # Almacenar el recuento de errores y el número total de muestras en el diccionario
    error_counts[unique_id] = (error_count, total_samples)

# Imprimir la proporción de veces que se cumple la condición sobre el número total de muestras para cada grupo
print(Color.CYAN + f"\nErrores máximos para convergencia [posición, alfa, beta, theta]: {POSERROR_THRESHOLD}m, {ORIERRROR_1_THRESHOLD}º, {ORIERRROR_2_THRESHOLD}º, {ORIERRROR_3_THRESHOLD}º" + Color.END)
print(Color.YELLOW + "\nNumero de veces que converge cada nube de puntos:" + Color.END)
for unique_id, (error_count, total_samples) in error_counts.items():
    if total_samples > 0:
        ratio = error_count / total_samples
        ratio_percent = int(ratio * 100)
        if ratio_percent <= 25:
            print(f"{Color.BOLD}Grupo {unique_id}:{Color.END}   {error_count} / {total_samples} = {Color.RED}{ratio_percent}%{Color.END}")
        elif ratio_percent > 25 and ratio_percent <= 75:
            print(f"{Color.BOLD}Grupo {unique_id}:{Color.END}   {error_count} / {total_samples} = {Color.YELLOW}{ratio_percent}%{Color.END}")
        else:
            print(f"{Color.BOLD}Grupo {unique_id}:{Color.END}   {error_count} / {total_samples} = {Color.GREEN}{ratio_percent}%{Color.END}")
    else:
        print(f"{Color.BOLD}Grupo {unique_id}:{Color.END}   No hay muestras")
