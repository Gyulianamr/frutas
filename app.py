import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

st.set_page_config(page_title="Clasificador de Frutas", page_icon="🍎")

@st.cache_resource
def cargar_modelo():
    modelo = tf.keras.models.load_model("models/modelo_frutas.h5")
    with open("models/clases.json") as f:
        clases = json.load(f)
    return modelo, clases

modelo, clases = cargar_modelo()

st.title("🍓 Clasificador de Frutas con IA")
st.caption("Proyecto por: Genesis Yuliana Medina Ramos")
st.write("Sube una imagen o toma una foto de una fruta y el modelo predecirá su categoría.")

tab1, tab2 = st.tabs(["📁 Subir imagen", "📷 Tomar foto"])

imagen_entrada = None

with tab1:
    archivo = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])
    if archivo is not None:
        imagen_entrada = Image.open(archivo).convert("RGB")

with tab2:
    foto = st.camera_input("Toma una foto de la fruta")
    if foto is not None:
        imagen_entrada = Image.open(foto).convert("RGB")

if imagen_entrada is not None:
    st.image(imagen_entrada, caption="Imagen a clasificar", use_column_width=True)

    img_resized = imagen_entrada.resize((100, 100))
    arr = np.array(img_resized) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = modelo.predict(arr)
    idx = np.argmax(pred)
    confianza = np.max(pred)

    st.success(f"Predicción: **{clases[idx]}**")
    st.write(f"Confianza: {confianza:.2%}")

st.markdown("---")
st.caption("Clasificador de frutas desarrollado con TensorFlow y Streamlit — Genesis Yuliana Medina Ramos")
