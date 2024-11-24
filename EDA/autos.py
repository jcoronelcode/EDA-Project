import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Se carga los datos
df = pd.read_csv('../autos.csv')

# Visualizamos la vista previa de los datos
print(df.head())
print(df.info())

# Estadísticas descriptivas de variables numéricas
print(df.describe())

# Valores únicos de variables categóricas
for col in ['brand', 'fuelType', 'vehicleType']:
    print(f"{col}: {df[col].unique()}")

# Busqueda de valores nulos
print(df.isnull().sum())

null_percentage = (df.isnull().sum() / len(df)) * 100
print(null_percentage)

# Imputar valores nulos en columnas categóricas
df['vehicleType'] = df['vehicleType'].fillna('unknown')
df['gearbox'] = df['gearbox'].fillna('unknown')
df['model'] = df['model'].fillna('unknown')
df['fuelType'] = df['fuelType'].fillna('unknown')
df['notRepairedDamage'] = df['notRepairedDamage'].fillna('no_informado')


# Columnas irrelevantes las eliminamos
df.drop(['nrOfPictures'], axis=1, inplace=True)

# Comprobamos los valores nulos tras la imputacion para verificar que no haya ninguno
print(df.isnull().sum())

# Definir función para calcular límites del IQR
def calcular_iqr(data, columna):
    Q1 = data[columna].quantile(0.25)
    Q3 = data[columna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    return limite_inferior, limite_superior

# Detectar outliers en 'price'
limite_inf_price, limite_sup_price = calcular_iqr(df, 'price')
outliers_price = df[(df['price'] < limite_inf_price) | (df['price'] > limite_sup_price)]

# Detectar outliers en 'kilometer'
limite_inf_km, limite_sup_km = calcular_iqr(df, 'kilometer')
outliers_km = df[(df['kilometer'] < limite_inf_km) | (df['kilometer'] > limite_sup_km)]

# Detectar outliers en 'yearOfRegistration'
limite_inf_year, limite_sup_year = calcular_iqr(df, 'yearOfRegistration')
outliers_year = df[(df['yearOfRegistration'] < limite_inf_year) | (df['yearOfRegistration'] > limite_sup_year)]


# Boxplots para visualizar outliers
plt.boxplot(df['price'], vert=False)
plt.title('Boxplot de Price')
plt.show()

plt.boxplot(df['kilometer'], vert=False)
plt.title('Boxplot de Kilometer')
plt.show()

plt.boxplot(df['yearOfRegistration'], vert=False)
plt.title('Boxplot de Year of Registration')
plt.show()

# Filtrar precios razonables (por ejemplo, >100 y <100,000)
df = df[(df['price'] > 100) & (df['price'] < 100000)]

# Filtrar años de registro razonables (1900 a 2024)
df = df[(df['yearOfRegistration'] >= 1900) & (df['yearOfRegistration'] <= 2024)]

# Filtrar kilometraje dentro de los límites del IQR
df = df[(df['kilometer'] >= limite_inf_km) & (df['kilometer'] <= limite_sup_km)]

# Comprobar que los valores están dentro de los límites establecidos
print(f"Price: Min = {df['price'].min()}, Max = {df['price'].max()}")
print(f"Year of Registration: Min = {df['yearOfRegistration'].min()}, Max = {df['yearOfRegistration'].max()}")
print(f"Kilometer: Min = {df['kilometer'].min()}, Max = {df['kilometer'].max()}")


# Histograma para la distribución de precios
plt.hist(df['price'], bins=50, color='blue', alpha=0.7)
plt.title('Distribución de Price')
plt.xlabel('Precio')
plt.ylabel('Frecuencia')
plt.show()

# Histograma para la distribución de kilometraje
plt.hist(df['kilometer'], bins=30, color='green', alpha=0.7)
plt.title('Distribución de Kilometer')
plt.xlabel('Kilometraje')
plt.ylabel('Frecuencia')
plt.show()

# Precio promedio por marca
precio_por_marca = df.groupby('brand')['price'].mean().sort_values(ascending=False)

# Gráfico de barras para precios promedio por marca
plt.figure(figsize=(10, 6))
precio_por_marca.plot(kind='bar', color='orange', alpha=0.8)
plt.title('Precio Promedio por Marca')
plt.xlabel('Marca')
plt.ylabel('Precio Promedio')
plt.xticks(rotation=45)
plt.show()

# Precio promedio por tipo de combustible
precio_por_combustible = df.groupby('fuelType')['price'].mean().sort_values(ascending=False)

# Gráfico de barras para precios promedio por tipo de combustible
plt.figure(figsize=(8, 5))
precio_por_combustible.plot(kind='bar', color='purple', alpha=0.8)
plt.title('Precio Promedio por Tipo de Combustible')
plt.xlabel('Tipo de Combustible')
plt.ylabel('Precio Promedio')
plt.xticks(rotation=45)
plt.show()

# Gráfico de cajas para comparar precios por tipo de transmisión
plt.figure(figsize=(8, 6))
df.boxplot(column='price', by='gearbox', grid=False, showfliers=True, patch_artist=True)
plt.title('Comparación de Precios por Transmisión')
plt.suptitle('')  # Elimina el título adicional por defecto
plt.xlabel('Tipo de Transmisión')
plt.ylabel('Precio')
plt.show()

# Gráfico de dispersión: Año de Registro vs Precio
plt.figure(figsize=(10, 6))
plt.scatter(df['yearOfRegistration'], df['price'], alpha=0.5, color='blue')
plt.title('Relación entre Año de Registro y Precio')
plt.xlabel('Año de Registro')
plt.ylabel('Precio')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Gráfico de dispersión: Kilometraje vs Precio
plt.figure(figsize=(10, 6))
plt.scatter(df['kilometer'], df['price'], alpha=0.5, color='green')
plt.title('Relación entre Kilometraje y Precio')
plt.xlabel('Kilometraje')
plt.ylabel('Precio')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# Seleccionar solo columnas numéricas
numerical_columns = df.select_dtypes(include=['int64', 'float64'])

# Calcular la matriz de correlación
correlation_matrix = numerical_columns.corr()

# Generar el mapa de calor
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title('Mapa de Calor de Correlación entre Variables Numéricas')
plt.show()