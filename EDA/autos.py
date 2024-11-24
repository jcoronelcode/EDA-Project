import pandas as pd

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
