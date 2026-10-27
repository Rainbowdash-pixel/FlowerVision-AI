import os
import json
import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image

# Configuration de la page
st.set_page_config(
    page_title="FlowerVision AI",
    page_icon="🌸",
    layout="wide"
)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

CLASSIFIER_PATH = os.path.join(
    MODEL_DIR, "mobilenetv2_flowers.keras"
)
SEGMENTER_PATH = os.path.join(
    MODEL_DIR, "unet_scratch.keras"
)
CLASSES_PATH = os.path.join(
    MODEL_DIR, "flower_class_names.json"
)

@st.cache_resource
def load_models():
    classifier = tf.keras.models.load_model(
        CLASSIFIER_PATH, compile=False
    )
    segmenter = tf.keras.models.load_model(
        SEGMENTER_PATH, compile=False
    )

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    return classifier, segmenter, class_names

st.title("🌸 FlowerVision AI")
st.subheader("Classification et segmentation de fleurs")
st.write(
    "Importez une photo pour obtenir une catégorie probable "
    "et visualiser la segmentation expérimentale de la fleur."
)

try:
    classifier, segmenter, class_names = load_models()
except Exception as e:
    st.error(f"Impossible de charger les modèles : {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choisissez une image de fleur",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Image originale", use_container_width=True)

    # Classification
    image_cls = image.resize((224, 224))
    array_cls = np.asarray(image_cls, dtype=np.float32)
    array_cls = tf.keras.applications.mobilenet_v2.preprocess_input(
        array_cls
    )
    array_cls = np.expand_dims(array_cls, axis=0)

    probabilities = classifier.predict(array_cls, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_name = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    with col2:
        st.markdown("### Résultat de classification")
        st.success(f"Catégorie prédite : **{predicted_name}**")
        st.metric("Confiance du modèle", f"{confidence * 100:.2f} %")

    # Segmentation
    st.markdown("---")
    st.markdown("### Résultat de segmentation")

    image_seg = image.resize((128, 128))
    array_seg = np.asarray(image_seg, dtype=np.float32) / 255.0
    array_seg = np.expand_dims(array_seg, axis=0)

    mask_probability = segmenter.predict(array_seg, verbose=0)[0, :, :, 0]
    mask_binary = mask_probability >= 0.5

    col3, col4 = st.columns(2)

    with col3:
        st.image(
            mask_binary.astype(np.uint8) * 255,
            caption="Masque binaire prédit",
            clamp=True,
            use_container_width=True
        )

    # Superposition du masque sur l'image
    original_small = np.asarray(image_seg).copy()
    overlay = original_small.copy()
    overlay[mask_binary] = (
        0.5 * overlay[mask_binary] + 0.5 * np.array([0, 255, 0])
    ).astype(np.uint8)

    with col4:
        st.image(
            overlay,
            caption="Superposition du masque sur la fleur",
            use_container_width=True
        )

    st.info(
        "La catégorie est une prédiction du modèle. "
        "Le masque est expérimental : il peut inclure des feuilles "
        "ou manquer certaines parties de la fleur."
    )

    # Trois prédictions les plus probables
    st.markdown("### Trois catégories les plus probables")
    top_indices = np.argsort(probabilities)[-3:][::-1]

    for idx in top_indices:
        st.write(
            f"**{class_names[int(idx)]}** — "
            f"{float(probabilities[idx]) * 100:.2f} %"
        )
