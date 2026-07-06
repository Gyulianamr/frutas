import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

st.set_page_config(page_title="Clasificador de Frutas", page_icon="🍎")

@st.cache_resource
def cargar_modelo():
    modelo = tf.keras.models.load_model("modelo_frutas.keras")
    with open("clases.json") as f:
        clases = json.load(f)
    return modelo, clases

modelo, clases = cargar_modelo()

st.title("🍓 Clasificador de Frutas con IA")
st.write("Sube una imagen de una fruta y el modelo predecirá su categoría.")

archivo = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

if archivo is not None:
    img = Image.open(archivo).convert("RGB")
    st.image(img, caption="Imagen cargada", use_column_width=True)

    img_resized = img.resize((100, 100))
    arr = np.array(img_resized) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = modelo.predict(arr)
    idx = np.argmax(pred)
    confianza = np.max(pred)

    st.success(f"Predicción: **{clases[idx]}**")
    st.write(f"Confianza: {confianza:.2%}")
