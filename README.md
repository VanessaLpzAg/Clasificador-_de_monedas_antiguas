# 🪙 **CLASIFICADOR DE MONEDAS ANTIGUAS** 🏛️

¡Bienvenido al repositorio del **Clasificador de Monedas Antiguas**! Este proyecto tiene como objetivo clasificar monedas históricas según su período temporal utilizando técnicas de Machine Learning. A continuación, te explico cómo está estructurado y qué encontrarás en cada sección.

---

## 📚 **Índice**

1. [Estructura del Proyecto](#-estructura-del-proyecto)
2. [Fuente de los Datos](#-fuente-de-los-datos)
3. [Objetivo del Proyecto](#-objetivo-del-proyecto)
4. [Importancia de las Columnas](#-importancia-de-las-columnas)
5. [Preprocesamiento de Datos](#-preprocesamiento-de-datos)
6. [Selección del Modelo](#-selección-del-modelo)
7. [Entrenamiento y Evaluación](#-entrenamiento-y-evaluación)
8. [Mejora e Implementación](#-mejora-e-implementación)
9. [Streamlit](#-streamlit)

---

## 📂 **Estructura del Proyecto**

```bash
|-- Clasificador_de_monedas_antiguas_ML
    |-- data
    |   |-- raw          
    |   |-- processed
    |
    |-- Notebooks
    |   |-- 01_Fuentes.ipynb
    |   |-- 02_LimpiezaEDA.ipynb
    |   |-- 03_Entrenamiento_Evaluacion.ipynb
        |-- data
        |-- img
    |
    |-- src
    |   |-- data_processing.py
    |   |-- training_evaluation.py       
    |
    |-- Models
    |   |-- Model_A.pkl
    |   |-- Model_grid_A.pkl
    |   |-- Model_B.pkl
    |   |-- Model_C.pkl
    |   |-- Model_D.pkl
    |   |-- Model_E.h5
    |   |-- Model_F.pkl
    |   |-- Final_Model_grid_A.pkl
    |   |-- Final_Model_config.yaml
    |
    |-- app_streamlit
    |   |-- app.py
    |   |-- requirements.txt
    |
    |-- docs
    |   |-- negocio.ppt
    |   |-- ds.ppt     
    |
    |-- README.md
```

---

## 📂 **Fuente de los Datos**

Los primeros cinco datasets fueron obtenidos del conjunto de datos 'Coin Images from the Portable Antiquities Scheme' disponible en Kaggle, subido por [Sarah Good](https://www.kaggle.com/segood). Posteriormente, se realizó web scraping en el [The British Museum](https://finds.org.uk/database) para obtener características adicionales de las monedas utilizadas para desarrollar este proyecto.

---

## 🎯 **Objetivo del Proyecto**

El objetivo principal es clasificar monedas antiguas en **5 períodos históricos** mediante sus características:

- Edad del Hierro (s.IX a.C. - s.I d.C.).
- Época romana (s.I - s.V d.C.).
- Época bajomedieval (s.V - s.XI d.C.).
- Época medieval (s. XI - XVI d.C.)
- Época postmedieval (s. XVI - s.XVII d.C.).

---

## 🔍 **Importancia de las Columnas**

- **Target (Período Histórico):** `broadperiod`
- **Características Físicas:** `thickness`, `diameter`, `weight`, `axis`, `axis_known`.
- **Características de Fabricación:** `Primary Material`, `Manufacture Method`.

---

## 🛠️ **Preprocesamiento de Datos**

1. **Limpieza de Datos:** Unificar dataset, eliminar valores nulos, duplicados.
2. **Codificación de Variables Categóricas:** Transformar variables categóricas en formatos numéricos.
3. **Balanceado del dataset:** Asegurar que todas las características estén en la misma escala.

---

## 🤖 **Selección del Modelo**

Se probarán varios algoritmos de clasificación (como HistGradient Boosting, Random Forest, SVM, Redes Neuronales, Redes Convolucionales y PCA.) para determinar cuál ofrece el mejor rendimiento.

---

## 🏋️ **Entrenamiento y Evaluación**

Se entrenará el modelo y se evaluará su rendimiento utilizando métricas como:

- **Accuracy**
- **Precisión**
- **Recall**
- **F1-Score**
- **Matriz de Confusión**

---

## 🚀 **Mejora e Implementación**

Después de la evaluación, se explorarán técnicas para mejorar el modelo, como:

- Ajuste de hiperparámetros.
- Selección de características.
- Uso de técnicas avanzadas como Grid Search o Cross-Validation.

---

## 💻 **Streamlit**

Para facilitar la visualización y el uso del modelo, se implementará una aplicación web utilizando **Streamlit**. Esta permitirá:

1. **Subir imágenes de monedas** para su clasificación.
2. **Ver los resultados de predicción** junto con su confiabilidad.
3. **Explorar métricas de evaluación** del modelo en tiempo real.

La aplicación se encuentra en la carpeta `app_streamlit/`, y para ejecutarla, usa el siguiente comando:

```bash
streamlit run app_streamlit/app.py
```
