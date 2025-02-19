
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from skimage.io import imread
import cv2
from tqdm import tqdm
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from tensorflow import keras
from keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score




# Importar DataFrame.
df_A = pd.read_csv('../data/processed/df_limpio.csv')

# Declarar las features y target.
X_A = df_A[['thickness', 'diameter', 'weight', 'axis', 'axis_known', 'Primary Material_n', 'Manufacture Method_n']]
y_A = df_A['period']

# Establecer las variables en train y en test.
XA_train, XA_test, yA_train, yA_test = train_test_split(X_A, y_A, test_size=0.2, random_state=7)

# Crear y entrenar el modelo A. 
model_A = HistGradientBoostingClassifier()
model_A.fit(XA_train, yA_train)

# Hacer predicciones
predicciones_A = model_A.predict(XA_test)

# Calcular las probabilidades de las predicciones.
predicciones_proba_A = model_A.predict_proba(XA_test)

# Realizar el cálculo de diferentes métricas.

# Accuracy score.
aciertos_A = accuracy_score(yA_test, predicciones_A)
errores_A = 1 - aciertos_A

# Varias métricas.
reporte_A = classification_report(yA_test, predicciones_A)
cm_A = confusion_matrix(yA_test, predicciones_A)

# Crear un pipeline para escalar los datos y hacer una validación cruzada.
clf_A = HistGradientBoostingClassifier()

pipeline_A = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', clf_A)
])

# Realizar validación cruzada
scores_A = cross_val_score(pipeline_A, X_A, y_A, cv=5, scoring='accuracy')

# Crear un pipeline para hacer una validación cruzada sin escalar los datos.

clf_A1 = HistGradientBoostingClassifier()

pipeline_A1 = Pipeline([    
    ('classifier', clf_A1)
])

# Realizar validación cruzada
scores_A1 = cross_val_score(pipeline_A1, X_A, y_A, cv=5, scoring='accuracy')

# Realizar un GridSearch con el modelo A.

# Parámetros.
param_grid_A = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_iter': [100, 200, 300],
    'max_leaf_nodes': [15, 31, 63],
    'min_samples_leaf': [10, 20, 30],
    'l2_regularization': [0, 1, 10]
}

gs_model_A = GridSearchCV(estimator=model_A, param_grid=param_grid_A, cv=3, scoring='accuracy', verbose=3)
gs_model_A.fit(XA_train, yA_train)

# Entrenar el modelo con los mejores parámetros del GridSearch.
param_grid_model_A = {
    'learning_rate': [0.05],
    'max_iter': [100],
    'max_leaf_nodes': [15],
    'min_samples_leaf': [30],
    'l2_regularization': [0]
}

grid_model_A = GridSearchCV(estimator=model_A, param_grid=param_grid_model_A, cv=3, scoring='accuracy', verbose=3)
grid_model_A.fit(XA_train, yA_train)

# Hacer predicciones.
predicciones_grid_A = grid_model_A.predict(XA_test)

# Calcular las probabilidades de las predicciones.
predicciones_proba_grid_A = grid_model_A.predict_proba(XA_test)

# Realizar el cálculo de diferentes métricas.

# Accuracy score.
aciertos_grid_A = accuracy_score(yA_test, predicciones_grid_A)
errores_grid_A = 1 - aciertos_A

# Varias métricas.
reporte_grid_A = classification_report(yA_test, predicciones_grid_A)

# Confusion_matrix.
cm_grid_A = confusion_matrix(yA_test, predicciones_grid_A)

# Guardar el Modelo A.

# Lo nombro 'ModeloA_HistGradientBoostingClassifier'.
with open('../Models/ModeloA_HistGradientBoostingClassifier.pkl', 'wb') as f:
    pickle.dump(model_A, f)

# Guardar el Modelo Grid A. (Es el mejor modelo.)

# Lo nombro 'Modelo_grid_A_HistGradientBoostingClassifier'.
with open('../Models/Modelo_grid_A_HistGradientBoostingClassifier.pkl', 'wb') as f:
    pickle.dump(grid_model_A, f)


# Establecer el modelo B.
model_B = RandomForestClassifier()

pipeline_B = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', model_B)
])

# Gridsearch del modelo B.
param_random_forest = {"classifier__n_estimators": [120],                      
                     "classifier__max_depth": [3,4,5,6,10,15,17],                       
                     "classifier__max_features": ["sqrt", 3, 4]                                                     
                     }

grid_model_B = GridSearchCV(pipeline_B, param_random_forest, cv=5, scoring='accuracy', n_jobs=-1)
grid_model_B.fit(XA_train, yA_train) 

# Hacer predicciones
predicciones_B = grid_model_B.predict(XA_test)

# Calcular las probabilidades de las predicciones.
predicciones_proba_B = grid_model_B.predict_proba(XA_test)

# Realizar el cálculo de diferentes métricas.

# Accuracy score.
aciertos_B = accuracy_score(yA_test, predicciones_B)
errores_B = 1 - aciertos_A

# Varias métricas.
reporte_B = classification_report(yA_test, predicciones_B)
# Confusion_matrix.
cm_B = confusion_matrix(yA_test, predicciones_B)

# Realizar validación cruzada
scores_B = cross_val_score(pipeline_B, X_A, y_A, cv=5, scoring='accuracy')

# Guardar el Modelo B.

# Lo nombro 'ModeloB_RandomForestClassifier'.
with open('../Models/ModeloB_RandomForestClassifier.pkl', 'wb') as f:
    pickle.dump(grid_model_B, f)

# Establecer el modelo C.
model_C = SVC()

pipeline_C = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', model_C)
])

# SVM.
param_svm = {"classifier__C": [0.01, 0.1, 1.0, 10.0], 
            "classifier__kernel": ["linear", "rbf"], 
            "classifier__degree": [2,3,4,5], 
            "classifier__gamma": ["scale", "auto"] 
           }

grid_model_C = GridSearchCV(pipeline_C, param_svm, cv=5, scoring='accuracy', n_jobs=-1)
grid_model_C.fit(XA_train, yA_train) 

# Hacer predicciones
predicciones_C = grid_model_C.predict(XA_test)

# Realizar el cálculo de diferentes métricas.

# Accuracy score.
aciertos_C = accuracy_score(yA_test, predicciones_C)
errores_C = 1 - aciertos_C

# Varias métricas.
reporte_C = classification_report(yA_test, predicciones_C)
# Confusion_matrix y su visualización.
cm_C = confusion_matrix(yA_test, predicciones_C)

# Realizar validación cruzada
scores_C = cross_val_score(pipeline_C, X_A, y_A, cv=5, scoring='accuracy')

# Guardar el Modelo C.

# Lo nombro 'ModeloC_SVC'.
with open('../Models/ModeloC_SVC.pkl', 'wb') as f:
    pickle.dump(grid_model_C, f)

# Crear el modelo D.

# Separar en train. test y validación.
XA_train_full, XA_test, yA_train_full, yA_test = train_test_split(X_A, y_A, test_size=0.2, random_state=7)

XA_train, XA_valid, yA_train, yA_valid = train_test_split(XA_train_full,
                                                         yA_train_full)

# Escalar los datos.
scaler = StandardScaler()
XA_train = scaler.fit_transform(XA_train)
XA_valid = scaler.transform(XA_valid)
XA_test = scaler.transform(XA_test)

# Establecer un modelo con cuatro capas y la de salida son 5 por las 5 épocas en las
# que se divide la clasificación.
model_D = keras.models.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(XA_train.shape[1],)),    
    keras.layers.Dense(64, activation='relu'),  
    keras.layers.Dense(32, activation='relu'),  
    keras.layers.Dense(16, activation='relu'),    
    keras.layers.Dense(5, activation='softmax')  
])

# Realizar la compilacion del modelo.
model_D.compile(
    loss='sparse_categorical_crossentropy',     # Al ser una clasificación multiclase.
    optimizer=keras.optimizers.RMSprop(learning_rate=0.0005), 
    metrics=['accuracy']
)

# Callbacks.
callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True), # Interrumpe el entrenamiento cuando no ve progreso en la validación.
    #keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6) # Reduce la tasa de aprendizaje si no hay mejora.
]

# Entrenar el modelo.
history = model_D.fit(
    XA_train,
    yA_train,
    epochs=100,
    batch_size=64,  
    validation_data=(XA_valid, yA_valid),
    callbacks=callbacks
)

model_D.summary()

# Hacer las predicciones.
predicciones_D = np.argmax(model_D.predict(XA_valid), axis=1)

# Realizar el cálculo de diferentes métricas.

# Accuracy score.
aciertos_D = accuracy_score(yA_valid, predicciones_D)
errores_D = 1 - aciertos_D

# Varias métricas.
reporte_D = classification_report(yA_valid, predicciones_D)

# Guardar el Modelo D.

# Lo nombro 'ModeloD_Redes_neuronales'.
with open('../Models/ModeloD_Redes_neuronales.pkl', 'wb') as f:
    pickle.dump(model_D, f)

# Modelo E.

# Definir las constantes.
IMAGE_WIDTH=32
IMAGE_HEIGHT=32
IMAGE_CHANNELS=3
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
BATCH_SIZE = 128
EPOCHS = 20

# Recorrer las diferentes imágenes y etiquetarlas.

IMAGENES = "../data/img/img_monedas_DL/"

# Lista de categorías y sus números correspondientes
category_mapping = {
    'Early Medieval': 0,
    'Iron Age': 1,
    'Medieval': 2,
    'Post medieval': 3,
    'Roman': 4
}

# Lista para almacenar los datos
data = []

# Iterar sobre las carpetas (categorías) en IMAGENES
for category in category_mapping:
    category_folder = os.path.join(IMAGENES, category)

    # Asegurarse de que la carpeta existe
    if os.path.isdir(category_folder):
        # Listar los archivos de imagen en la carpeta de la categoría
        image_files = [f for f in os.listdir(category_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # Agregar los datos al DataFrame
        for image in image_files:
            data.append([image, category, category_mapping[category]])

# Crear un DataFrame con los datos
df = pd.DataFrame(data, columns=['Nombre', 'Periodo', 'Periodo_num'])

# Resize y convertir a blanco y negro cada imagen.

# Definir el path de las imágenes
IMAGENES = "../data/img/img_monedas_DL/"

# Lista de categorías y sus números correspondientes
category_mapping = {
    'Early Medieval': 0,
    'Iron Age': 1,
    'Medieval': 2,
    'Post medieval': 3,
    'Roman': 4
}

# Arrays para las imágenes y categorías
X = []
y = []

# Iterar sobre las carpetas (categorías)
for category in category_mapping:
    category_folder = os.path.join(IMAGENES, category)

    print(category)

    if os.path.isdir(category_folder):
        # Listar los archivos de imagen en la carpeta de la categoría
        image_files = [f for f in os.listdir(category_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Procesar todas las imágenes en la carpeta
        for image_file in tqdm(image_files, desc=f'Procesando {category}'):
            # Construir la ruta completa de la imagen
            image_path = os.path.join(category_folder, image_file)
            
            # Leer la imagen en blanco y negro
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            # Verificar si la imagen se ha cargado correctamente
            if image is None:
                print(f"Error al cargar la imagen: {image_path}")
                continue
            
            # Redimensionar la imagen
            imagesmall = cv2.resize(image, (128, 128))
            
            # Normalizar los píxeles a [0, 1]
            imagesmall = imagesmall / 255.0
            
            # Expandir las dimensiones para tener el canal (1 canal para B/N)
            imagesmall = np.expand_dims(imagesmall, axis=-1)
            
            # Agregar a la lista de imágenes y categorías
            
            if category == 'Early Medieval':
                category_num = 0
            elif category == 'Iron Age':
                category_num = 1
            elif category == 'Medieval':
                category_num = 2
            elif category == 'Post medieval':
                category_num = 3
            elif category == 'Roman':
                category_num = 4
            
            X.append(imagesmall)
            y.append(category_num)
            

# Convertir a arrays de NumPy
X = np.array(X)
y = np.array(y)

# Desordeno las imagenes ya que están agrupadas por época.
X, y = shuffle(X, y, random_state=7)

# Divido las imágenes en train y test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Guardar los arrays de las imágenes.
CARPETA = '../data/img/Arrays_imagenes'

np.savez(CARPETA + '/data.npz',
        X_train = X_train,
        y_train = y_train,
        X_test = X_test,
        y_test = y_test)

# Volver a cargarlos.
data = np.load(CARPETA + '/data.npz')
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

# Crear el modelo E.

layers = [
    keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(128, 128, 1)),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    
    keras.layers.Conv2D(32, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),

    keras.layers.Conv2D(32, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    
    keras.layers.Flatten(),    
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(5, activation='softmax')
]

model_E = keras.Sequential(layers)

model_E.compile(optimizer = 'adam',
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])

# Ver el desarrollo del modelo.
model_E.summary()

# Crear un Early Stopping para entrenar el modelo.
earlystop = EarlyStopping(patience=5)

# Entrenar el modelo E.
history_E = model_E.fit(X_train,
         y_train,
         epochs = 20,
         batch_size = BATCH_SIZE,
         callbacks = [earlystop],
         validation_split = 0.2)

# Evaluación modelo E.
results = model_E.evaluate(X_test, y_test)

# Generar predicciones
y_pred = model_E.predict(X_test)
y_pred_class = y_pred.argmax(axis=-1)  

# Calcular métricas.'''

# Matriz de confusión.
cm_E = confusion_matrix(y_test, y_pred_class)

# Comprobar modelo con una imagen externa.

# Cargar y procesar la imagen externa en escala de grises
img_path = '../data/img/img_monedas_predecir/Marco Aurelio.png'

# Cargar la imagen en escala de grises
img = Image.open(img_path).convert('L') 

# Redimensionar la imagen a las dimensiones requeridas por el modelo
img = img.resize((128, 128))  

# Convertir la imagen a un array de numpy
img_array = np.array(img)

# Normalizar la imagen si el modelo lo requiere
img_array = img_array / 255.0 

# Asegurarse de que la imagen tenga la forma correcta (batch_size, height, width, channels)
img_array = np.expand_dims(img_array, axis=-1)  

# Añadir una dimensión adicional para el batch_size
img_array = np.expand_dims(img_array, axis=0)  # Forma (1, 128, 128, 1)

# Realizar la predicción.
pred = model_E.predict(img_array)

# Interpretar la predicción.
predicted_class = np.argmax(pred, axis=-1) 

# Guardar el Modelo E.

# Lo nombro 'ModeloE_Redes_Convolucionales'.
with open('../Models/ModeloE_Redes_Convolucionales.pkl', 'wb') as f:
    pickle.dump(model_E, f)

# Establecer el modelo F.

# Definir el pipeline con estandarización, reducción de dimensionalidad y modelo de clasificación
pipeline_F = Pipeline([
    ('scaler', StandardScaler()),       
    ('pca', PCA(n_components=7)),        
    ('rf', RandomForestClassifier())  
])

# Definir los hiperparámetros a buscar en GridSearchCV.
param_grid_F = {
    'pca__n_components': [4, 5, 6, 7],         
    'rf__n_estimators': [100, 200],              
    'rf__max_depth': [5, 10, 20],             
    'rf__min_samples_split': [2, 5, 10]               
}

# Configurar GridSearchCV.
grid_search_F = GridSearchCV(
    estimator=pipeline_F,
    param_grid=param_grid_F,
    cv=5,                   
    scoring='accuracy',     
    n_jobs=-1               
)

# Establecer las variables en train y en test.
XA_train, XA_test, yA_train, yA_test = train_test_split(X_A, y_A, test_size=0.2, random_state=7)

# Entrenar el modelo.
grid_search_F.fit(XA_train, yA_train)

# Mejores parámetros encontrados.
grid_search_F.best_params_

# Predecir en el conjunto de prueba.
predicciones_F = grid_search_F.predict(XA_test)

# Evaluar el modelo.

# Accuracy.
aciertos_F = accuracy_score(yA_test, predicciones_F)
errores_F = 1 - aciertos_F

# Varias métricas.
reporte_F = classification_report(yA_test, predicciones_F)

# Confusion_matrix.
cm_F = confusion_matrix(yA_test, predicciones_F)

# Guardar el Modelo F.

# Lo nombro 'ModeloF_PCA_RF'.
with open('../Models/ModeloF_PCA_RF.pkl', 'wb') as f:
    pickle.dump(grid_search_F, f)


