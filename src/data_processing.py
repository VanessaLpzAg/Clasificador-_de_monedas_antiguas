import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder
import os
import shutil

# Importar el dataset creado con los resultados del webscrapping.
df_webscrap = pd.read_csv('../data/raw/Resultados_Webscrapping.csv')

# Importar el dataset descargados en Kaggle.
df_original = pd.read_csv('../data/processed/df_2.csv')

# Unificar los dos dataframe a tráves de un merge.
# Cambiar el nombre de la columna del df_original a Unique ID.
df_original.rename(columns={'old_findID' : 'Unique ID'}, inplace= True)

# Unir df_original y df_webscrap usando la columna 'Unique ID'
df_final = pd.merge(df_original, df_webscrap, on='Unique ID', how='inner')

#  Al existir solo un registro, procedo a eliminarlo, ya que no es representativo.
df_final = df_final.drop(df_final[df_final['broadperiod'] == 'BYZANTINE'].index)

# Unificar las monedas romanas en un periodo romano, aunque su procedencia sea provincial, ya que nuestro proyecto consiste
# en su clasificación según la época y no procedencia.
df_final['broadperiod'] = df_final[ 'broadperiod']. replace({'GREEK AND ROMAN PROVINCIAL' : 'ROMAN'})

# Reconvertir la variable Target en numérica, por haber estandarizado en dos periodos menos.
# Crear una instancia de LabelEncoder
convertidor = LabelEncoder()

# Aplicar la codificación a la columna 'broadperiod'
df_final['period'] = convertidor.fit_transform(df_final['broadperiod'])

# Borrar columnas.
df_final = df_final.drop(columns= ['id', 'uri', 'secuid', 'objecttype', 'created', 'updated', 'institution', 'creator', 'discoveryMethod', 'finder',
                                'imagedir', 'thumbnail', 'currentLocation', 'subsequentActionTerm', 'gridSource','fourFigure', 'knownas',
                                'parish', 'filename', 'county', 'datefound1', 'datefound2', 'typeTerm', 'notes', 'reasonTerm', 'note',
                                'description', 'subperiodTo', 'periodFromName', 'todate', 'fromdate', 'TID', 'musaccno', 'smrRef', 'otherRef',
                                'quantity', 'objecttype', 'classification', 'subClassification', 'cciNumber', 'mintmark', 'geography',
                                'reeceID', 'periodToName','district','subperiodFrom'], axis = 1)


# Al haber muchos nulls y además de la época medieval hay bastantes monedas, voy a balancear el dataset y a la vez bajar el % de nulls.

# Paso 1: Separar las clases
df_medieval = df_final[df_final['broadperiod'] == 'MEDIEVAL']
df_post_medieval = df_final[df_final['broadperiod'] == 'POST MEDIEVAL']

# Paso 2: Obtener el tamaño de la clase menor (POST MEDIEVAL)
min_class_len = len(df_post_medieval)  # Tamaño de la clase menor

# Paso 3: Tomar una muestra aleatoria de la clase mayoritaria (MEDIEVAL) para que tenga el mismo tamaño que la clase menor
df_medieval_balanced = df_medieval.sample(n=min_class_len, random_state=7)

# Paso 4: Concatenar las clases balanceadas
df_balanceado = pd.concat([
    df_medieval_balanced,
    df_post_medieval,
])

# Borro del dataset inicial las monedas medievales y postmedievales para evitar duplicados cuando de unifique la parte balanceada con la original.

# Borro del dataset las monedas que sean Medieval.
df_final = df_final.drop(df_final[df_final['broadperiod'] == 'MEDIEVAL'].index)

#  Borro del dataset las monedas que sean PostMedieval.
df_final = df_final.drop(df_final[df_final['broadperiod'] == 'POST MEDIEVAL'].index)

# Unificar los dos dataset, el balanceado y el final.
df_final_1 = pd.concat([df_balanceado, df_final])

# Definir el nuevo orden de las columnas
nuevo_orden = ['Unique ID', 'broadperiod', 'period', 'thickness', 'diameter', 'weight', 'axis', 'Primary material', 
                'Manufacture method', 'URL', 'Image URL', 'Saved Image','length', 
               'height', 'width', 'obverseDescription', 'reverseDescription', 'obverseLegend', 'reverseLegend', 'reverseType',
               'regionName', 'cultureName', 'inscription', 'objectCertainty', 
               'rulerName', 'mintName', 'denominationName', 'tribeName', 'categoryTerm', 'moneyerName']

# Reordenar el DataFrame
df_limpio = df_final_1[nuevo_orden]

# Al renombrar el dataframe, realizo una copia para asegurar los futuros cambios.
df_limpio = df_limpio.copy()

# Convertir la columna Primary material en numérica.

# Crear una instancia de LabelEncoder
convertidor = LabelEncoder()

# Aplicar la codificación a la columna 'Primary material'
df_limpio.loc[:, 'Primary Material_n'] = convertidor.fit_transform(df_limpio['Primary material'])

# Convertir la columna Manufacture method en numérica.

# Crear una instancia de LabelEncoder
convertidor = LabelEncoder()

# Aplicar la codificación a la columna 'Manufacture method'
df_limpio.loc[:, 'Manufacture Method_n'] = convertidor.fit_transform(df_limpio['Manufacture method'])

# Convertir los Nan de la columna en valor -1 (fuera de rango) y a la vez una columna para indicar si es un dato conocido o desconocido.

# Reemplazar los NaN en 'axis' con -1.
df_limpio['axis'] = df_limpio['axis'].fillna(-1)

# Crear la columna dummy indicando si 'axis' era conocido (1) o desconocido (0)
df_limpio.loc[:,'axis_known'] = (df_limpio['axis'] != -1).astype(int)

# Columna thickness.

# Calcular la mediana, media y moda según su época.
print(df_limpio.groupby('broadperiod')['thickness'].transform('mean').round(2).value_counts())
print(df_limpio.groupby('broadperiod')['thickness'].transform(lambda x: x.mode()[0]).value_counts())
print(df_limpio.groupby('broadperiod')['thickness'].transform('median').round(2).value_counts())

# Al comprobar que hay bastantes outliers tomo la decisión de sustituirlo por la mediana.

# Guardar la mediana de grosor por periodo.
mediana_grosor_por_epoca = df_limpio.groupby('broadperiod')['thickness'].transform('median')

# Imputar los nulos con la mediana de su periodo.
df_limpio['thickness'] = df_limpio['thickness'].fillna(mediana_grosor_por_epoca)

#Columna diameter.

# Calcular la mediana, media y moda según su época.
print(df_limpio.groupby('broadperiod')['diameter'].transform('mean').round(2).value_counts())
print(df_limpio.groupby('broadperiod')['diameter'].transform(lambda x: x.mode()[0]).value_counts())
print(df_limpio.groupby('broadperiod')['diameter'].transform('median').round(2).value_counts())

# Al comprobar que hay bastantes outliers tomo la decisión de sustituirlo por la mediana.

# Guardar la mediana de grosor por periodo.
mediana_diametro_por_epoca = df_limpio.groupby('broadperiod')['diameter'].transform('median')

# Imputar los nulos con la mediana de su periodo.
df_limpio['diameter'] = df_limpio['diameter'].fillna(mediana_diametro_por_epoca)

# Columna weight.

# Calcular la mediana, media y moda según su época.
print(df_limpio.groupby('broadperiod')['weight'].transform('mean').round(2).value_counts())
print(df_limpio.groupby('broadperiod')['weight'].transform(lambda x: x.mode()[0]).value_counts())
print(df_limpio.groupby('broadperiod')['weight'].transform('median').round(2).value_counts())

# Al comprobar que hay bastantes outliers tomo la decisión de sustituirlo por la mediana.

# Guardar la mediana de grosor por periodo.
mediana_peso_por_epoca = df_limpio.groupby('broadperiod')['weight'].transform('median')

# Imputar los nulos con la mediana de su periodo.
df_limpio['weight'] = df_limpio['weight'].fillna(mediana_peso_por_epoca)

# Guardar el dataset limpio.
df_limpio.to_csv('../data/processed/df_limpio.csv', index=False)

# Extraer las fotos correspondiente al dataframe y guardarlas en una carpeta.

# Ruta de la carpeta con las fotos y la carpeta de destino
carpeta_origen = '../data/img/Fotos_monedas'
carpeta_destino = '../data/img/img_monedas_DL'

# Cargar el DataFrame
df = pd.read_csv('../data/processed/df_limpio.csv')
ids_df = df['Unique ID'].astype(str).tolist()  # Convierte a str por si acaso

# Obtener las fotos en la carpeta de origen
fotos_carpeta = [f for f in os.listdir(carpeta_origen) if os.path.isfile(os.path.join(carpeta_origen, f))]
ids_carpeta = [os.path.splitext(f)[0] for f in fotos_carpeta]  # Quita la extensión

# Comparar y mover fotos coincidentes
for foto in fotos_carpeta:
    id_foto = os.path.splitext(foto)[0]  # ID sin extensión
    if id_foto in ids_df:
        ruta_origen = os.path.join(carpeta_origen, foto)
        ruta_destino = os.path.join(carpeta_destino, foto)
        shutil.move(ruta_origen, ruta_destino)
        print(f'Movida: {foto}') 

# Guardar y clasificar en carpetas las diferentes fotografías de monedas por época, para poder utilizarlas en modelos Deep Learning.

# Establecer el DataFrame
df = df_limpio

# Ruta de la carpeta con las imágenes
carpeta_imagenes = '../data/img/img_monedas_DL'

# Crear una lista de épocas (basada en broadperiod)
epocas = ['EARLY MEDIEVAL', 'IRON AGE', 'MEDIEVAL', 'POST MEDIEVAL', 'ROMAN']

# Crear carpetas para cada época si no existen
for epoca in epocas:
    ruta_carpeta = os.path.join(carpeta_imagenes, epoca)
    if not os.path.exists(ruta_carpeta):
        os.makedirs(ruta_carpeta)

# Recorrer cada imagen en el DataFrame
for index, row in df.iterrows():
    nombre_imagen = str(row['Saved Image'])  
    nombre_imagen = nombre_imagen.split('\\')[-1] 
    epoca = row['broadperiod']  

    # Verificar si la época es válida
    if epoca in epocas:
        # Verificar si la imagen existe en la carpeta origen
        ruta_origen = os.path.join(carpeta_imagenes, nombre_imagen)
        if os.path.exists(ruta_origen):
            # Definir la ruta de destino según la época
            ruta_destino = os.path.join(carpeta_imagenes, epoca, nombre_imagen)
            # Mover la imagen a su carpeta correspondiente
            shutil.move(ruta_origen, ruta_destino)
            print(f'Moviendo {nombre_imagen} a {epoca}')
        else:
            print(f'Imagen no encontrada: {nombre_imagen}')
    else:
        print(f'Época no válida para {nombre_imagen}: {epoca}')


