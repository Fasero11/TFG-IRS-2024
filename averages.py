import pandas as pd
import os

# Obtener la ruta del directorio HOME
home_route = os.path.expanduser("~")

# Definir la ruta del archivo CSV
filename = "errordata.csv"
filepath = os.path.join(home_route, filename)

# Cargar el archivo CSV en un DataFrame de pandas
data = pd.read_csv(filepath)

# Calcular la media, mediana, mínimo, máximo y la desviación estándar de cada columna
means = data.mean()
medians = data.median()
minimums = data.min()
maximums = data.max()
std_devs = data.std()

# Contar el número de filas (descartando el encabezado)
num_rows = data.shape[0] - 1

# Imprimir las estadísticas de cada columna
print("\n")
print(f"{'Columna':<12} {'Media':<12} {'Mediana':<12} {'Mínimo':<12} {'Máximo':<12} {'Desviación estándar':<20}")
print("-------------------------------------------------------------------------------------------------")
for column in data.columns[7:]:
    print(f"{column:<12} {round(means[column], 4):<12} {round(medians[column], 4):<12} {round(minimums[column], 4):<12} {round(maximums[column], 4):<12} {round(std_devs[column], 4):<20}")

# Imprimir el número de filas
print(f"\nNúmero de muestras: {num_rows}\n")
