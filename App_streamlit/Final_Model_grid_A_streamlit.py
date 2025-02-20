import streamlit as st
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

# Configuración de la página
st.set_page_config(
    page_title='Clasificación de Monedas Antiguas',
    page_icon='🪙',
    layout='wide'
)

# CSS personalizado para el fondo y los colores de los sliders y pestañas
st.markdown(
    """
    <style>
    .stApp {
        background-color: #01001b;        
    }
    /* Cambiar el color de los sliders */
    .stSlider .st-eb {
        background-color: #679bfe !important;
    }
    .stSlider .st-ec {
        background-color: #679bfe !important;
    }
    .stSlider .st-ed {
        background-color: #679bfe !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

NB = st.slider


ColorMinMax = st.markdown(''' <style> div.stSlider > div[data-baseweb = "slider"] > div[data-testid="stTickBar"] > div {
    background: rgb(1 1 1 / 0%); } </style>''', unsafe_allow_html = True)


Slider_Cursor = st.markdown(''' <style> div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"]{
    background-color: rgb(103, 155, 254); box-shadow: rgb(14 38 74 / 20%) 0px 0px 0px 0.2rem;} </style>''', unsafe_allow_html = True)

    
Slider_Number = st.markdown(''' <style> div.stSlider > div[data-baseweb="slider"] > div > div > div > div
                                { color: rgb(103, 155, 254); } </style>''', unsafe_allow_html = True)
    

# Cargar la plantilla de título y descripción.
st.markdown('''<div style="position: relative; width: 100%; height: 0; padding-top: 25.0000%;
 padding-bottom: 0; box-shadow: 0 2px 8px 0 rgba(63,69,81,0.16); margin-top: 1.6em; margin-bottom: 0.9em; overflow: hidden;
 border-radius: 8px; will-change: transform;">
  <iframe loading="lazy" style="position: absolute; width: 100%; height: 100%; top: 0; left: 0; border: none; padding: 0;margin: 0;"
    src="https://www.canva.com/design/DAGfjmrI0cc/8uuiX4Q_qH2x-TtwDTvHSw/watch?embed" allowfullscreen="allowfullscreen" allow="fullscreen">
  </iframe>
</div>''', unsafe_allow_html=True) 

# Cargar el modelo guardado para el clasificador basado en características
with open('../Models/Final_Model_grid_A_HistGradientBoostingClassifier.pkl', 'rb') as f:
    modelo = pickle.load(f)

# Cargar el modelo convolucional
cnn_model = tf.keras.models.load_model('../Models/ModeloE_Redes_Convolucionales.h5')  # Actualiza la ruta

# CSS para modificar las pestañas
st.markdown(
    """
    <style>
    div[data-baseweb="tab-list"] button {
        font-size: 22px !important; /* Aumenta el tamaño */
        font-weight: bold !important; /* Negrita */
        padding: 12px 24px !important; /* Ajuste de espaciado */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Crear pestañas
tab1, tab2 = st.tabs(["📜 Clasificación por características", "🖼️ Clasificación por Imagen"])

# Pestaña 1: Clasificación por características
with tab1:
    
        st.markdown(
    "<h3 style='color: #CCCCCC; text-align: center;'>Características de la moneda:</h3>", 
    unsafe_allow_html=True
)
        
        # Diccionarios
        material_dict = {
            'Base de plata': 0,
            'Cobre': 1,
            'Aleación de cobre': 2,
            'Oro': 3,
            'Hierro': 4,
            'Plomo': 5,
            'Aleación de plomo': 6,
            'Otros': 7,
            'Plata': 8,
            'Estaño o aleación de estaño': 9,
            'Metal blanco': 10
        }

        manufactura_dict = {
            'Fundido': 0,
            'Fabricado': 1,
            'Fresado': 2,
            'Múltiple': 3,
            'Golpeado o martillado': 4,
            'Rueda fabricada': 5,
            'Indeterminado': 6
        }

        with st.container():

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            Thickness = st.slider('Grosor (mm)', min_value=0.05, max_value=75.0, value=10.0, step=0.01)
            Diameter = st.slider('Diámetro (mm)', min_value=0.4, max_value=132.0, value=70.0, step=0.1)
            Weight = st.slider('Peso (g)', min_value=0.1, max_value=587.0, value=100.0, step=0.1)
            Axis = st.slider('Axis', min_value=-1.0, max_value=255.0, value=-1.0, step=0.1)
            Axis_conocido = 0 if Axis == -1 else 1

            Axis_conocido_text = "Desconocido" if Axis_conocido == 0 else "Conocido"
            st.markdown(f"<p>Axis_conocido: <b>{Axis_conocido_text}</b></p>", unsafe_allow_html=True)

            Material = st.selectbox('Materia prima', list(material_dict.keys()))
            Material_n = material_dict[Material]
            Manufactura = st.selectbox('Método de Manufactura', list(manufactura_dict.keys()))
            Manufactura_n = manufactura_dict[Manufactura]
                
            st.markdown("</div>", unsafe_allow_html=True)

            if st.button('Predecir'):
                input_data = np.array([[Thickness, Diameter, Weight, Axis, Axis_conocido, Material_n, Manufactura_n]])
                prediccion_proba = modelo.predict_proba(input_data)  # Probabilidades de cada clase
                prediccion = modelo.predict(input_data)

                resultado = {
                    0: 'La moneda es de época bajomedieval (s.V - XI d.C)',
                    1: 'La moneda es de la Edad del Hierro (s.IX a.C - I d.C)',
                    2: 'La moneda es de época medieval (s.XI - XVI d.C)',
                    3: 'La moneda es de época postmedieval (s.XVI - XVII d.C)',
                    4: 'La moneda es de época romana (s.I - V d.C)',
                }

                # Obtener el porcentaje de acierto
                probabilidad = np.max(prediccion_proba) * 100
                prediccion_texto = resultado.get(prediccion[0], 'Época desconocida.')

                # Mostrar la predicción y la confianza en recuadros separados
                st.markdown(
                f"""
                <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
                    <div style="background-color: #B8860B; padding: 15px; border-radius: 10px; text-align: center;">
                        <h3 style="color: white;">{prediccion_texto}</h3>
                    </div>
                    <div style="background-color: #D2B48C; padding: 10px; border-radius: 10px; text-align: center;">
                        <h4 style="color: white;">Confianza: {probabilidad:.2f}%</h4>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
    )

# Pestaña 2: Clasificación por Imagen
with tab2:
    st.markdown("<h3 style='color: #CCCCCC;'>Sube una imagen para clasificar la moneda:</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Selecciona una imagen de la moneda", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Mostrar la imagen cargada
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen de la moneda", use_container_width=True)

        # Preprocesar la imagen para el modelo convolucional
        # Convertir la imagen a escala de grises
        image = image.convert('L')  # 'L' es el modo para escala de grises
        image = image.resize((128, 128))  # Ajusta el tamaño según el modelo
        image = np.array(image) / 255.0  # Normalizar
        image = np.expand_dims(image, axis=0)  # Asegurar que tenga el formato adecuado

        if st.button('Predecir Época'):
            prediction = cnn_model.predict(image)
            class_idx = np.argmax(prediction, axis=1)[0]
            probabilidad_imagen = np.max(prediction) * 100  # Obtener la probabilidad más alta

            resultado_imagen = {
                0: 'La moneda es de época bajomedieval (s.V - XI d.C)',
                1: 'La moneda es de la Edad del Hierro (s.IX a.C - I d.C)',
                2: 'La moneda es de época medieval (s.XI - XVI d.C)',
                3: 'La moneda es de época postmedieval (s.XVI - XVII d.C)',
                4: 'La moneda es de época romana (s.I - V d.C)',
            }

            # Mostrar los resultados en recuadros separados
            st.markdown(
                f"""
                <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
                    <div style="background-color: #B8860B; padding: 15px; border-radius: 10px; text-align: center;">
                        <h3 style="color: white;">{resultado_imagen.get(class_idx, 'Época desconocida.')}</h3>
                    </div>
                    <div style="background-color: #D2B48C; padding: 10px; border-radius: 10px; text-align: center;">
                        <h4 style="color: white;">Confianza: {probabilidad_imagen:.2f}%</h4>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
    )