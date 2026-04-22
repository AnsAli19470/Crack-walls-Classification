"""
Streamlit UI: upload an image and classify crack vs non-crack using the trained
Keras model.

This app mirrors the user's working Colab inference:
    - resize to 128x128
    - convert to array
    - expand batch dimension
    - call model.predict(...)
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.utils import img_to_array, load_img

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "best_classifier_model.keras"
IMG_SIZE = (128, 128)
CLASS_NAMES = ("Crack", "Non Crack")


@st.cache_resource
def load_model() -> tf.keras.Model:
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_FILE}")
    return tf.keras.models.load_model(MODEL_FILE)


def preprocess_image(uploaded_bytes: bytes) -> np.ndarray:
    """
    Match the user's Colab inference exactly: resize and batch the raw RGB image.
    """
    img = load_img(io.BytesIO(uploaded_bytes), target_size=IMG_SIZE)
    arr = img_to_array(img)
    return np.expand_dims(arr, axis=0)


def main() -> None:
    st.set_page_config(
        page_title="Crack vs Non-crack",
        page_icon="🔍",
        layout="centered",
    )
    st.title("Crack vs non-crack classifier")

    try:
        model = load_model()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()

    uploaded = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
    )

    if not uploaded:
        st.info("Upload an image, then click **Predict** to run the model.")
        return

    data = uploaded.getvalue()
    pil = Image.open(io.BytesIO(data))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        st.image(pil, width="stretch")

    with col2:
        st.subheader("Prediction")
        predict_clicked = st.button("Predict", type="primary", width="stretch")

        if predict_clicked:
            batch = preprocess_image(data)
            prob_raw = float(model.predict(batch, verbose=0)[0][0])
            st.session_state["prob_raw"] = prob_raw
            st.session_state["pred_file_id"] = uploaded.file_id

        has_result = (
            st.session_state.get("pred_file_id") == uploaded.file_id
            and "prob_raw" in st.session_state
        )

        if has_result:
            prob_class_1 = float(st.session_state["prob_raw"])
            prob_class_0 = 1.0 - prob_class_1
            if prob_class_1 > 0.5:
                label = CLASS_NAMES[1]
                confidence = prob_class_1
            else:
                label = CLASS_NAMES[0]
                confidence = prob_class_0

            st.metric("Prediction", label)
            st.metric("Confidence (this class)", f"{confidence * 100:.1f}%")
            st.progress(float(confidence), text=f"{label}: {confidence * 100:.1f}%")

            with st.expander("All scores"):
                st.write(
                    {
                        f"Raw sigmoid ({CLASS_NAMES[1]})": f"{prob_class_1 * 100:.2f}%",
                        f"P({CLASS_NAMES[0]})": f"{prob_class_0 * 100:.2f}%",
                        f"P({CLASS_NAMES[1]})": f"{prob_class_1 * 100:.2f}%",
                    }
                )
        elif not predict_clicked:
            st.caption("Click **Predict** to classify this image.")


if __name__ == "__main__":
    main()
