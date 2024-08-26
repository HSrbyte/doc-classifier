import cv2
import time
import numpy as np
import pandas as pd
import streamlit as st

import tensorflow as tf
from tensorflow.keras.models import load_model

from typing import List, Dict, Optional
from src import (TextModel, inference, create_inference_dataset,
                draw_tesseract_result, image_rotate)


default_categories: Dict[int, str] = {
        0: "email",
        1: "handwritten",
        2: "invoice",
        3: "national_identity_card",
        4: "passport",
        5: "scientific_publication"
    }

layers_gradcam: Dict[str, str] = {
    "CNN": "conv2d_2",
    "SqueezNet": "conv2d_25",
    "MobileNetV2": "Conv_1_bn",
    "EfficientNet": "top_conv",
    "ResNet50": "conv5_block3_out",
}

base_models: Dict[str, Optional[str]] = {
    "CNN": "conv2d_2",
    "SqueezNet": None,
    "MobileNetV2": None,
    "EfficientNet": "efficientnetb1",
    "ResNet50": "resnet50",
}

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def load_text_model(text_methode: str, text_model_name: str) -> TextModel:

    dict_text_methode = {
        "Text": "tfidfOnly",
        "Structure": "words_scaled",
        "Text & Structure" : "words_structure"
    }
    methode = dict_text_methode[text_methode]
    text_model_path = f"models/{methode}_{text_model_name}.joblib"

    if methode == "tfidfOnly" or methode == "words_structure":
        tfidf = "models/TfidfVectorizer.joblib"
    else:
        tfidf = None

    if methode == "words_scaled" or methode == "words_structure":
        standard_scaler = "models/StandardScaler.joblib"
    else:
        standard_scaler = None

    common_words = "data/processed/most_common_words.json"
    return TextModel(text_model_path, tfidf, standard_scaler, common_words)


def load_image_model(model_path: str, device: str) -> tf.keras.models:
    with tf.device(device):
        model = load_model(model_path)
    return model


def st_plot(result_text: Optional[List[float]] = None,
            result_image: Optional[List[float]] = None,
            text_model_name: Optional[str] = None,
            image_model_name: Optional[str] = None
            ) -> None:

    data = {}
    data['Label'] = ["email", "handwritten", "invoice", "national_identity_card",
                     "passport", "scientific_publication"]
    if result_text is not None:
        _text_key = 'Text Model' if text_model_name is None else f"Text Model ({text_model_name})"
        data[_text_key] = result_text
    if result_image is not None:
        _image_key = 'Image Model' if image_model_name is None else f"Image Model ({image_model_name})"
        data[_image_key] = result_image[0]
    data = pd.DataFrame(data)
    st.bar_chart(data, x="Label", stack=False, horizontal=True)


def compute_gradcam(img, model, class_idx):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = model(img)
        loss = predictions[:, class_idx]

    # Gradient of the loss with respect to the convolutional layer output
    grads = tape.gradient(loss, conv_outputs)

    # Compute the guided gradients
    guided_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the convolutional layer output with the computed gradients
    conv_outputs = conv_outputs[0]
    guided_grads = guided_grads[None, None, :]
    weighted_conv_outputs = conv_outputs * guided_grads

    # Compute the Grad-CAM heatmap
    cam = np.mean(weighted_conv_outputs, axis=-1)

    # Normalize the heatmap
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    return cam


def grad_cam(img: np.ndarray, layer_name: str, saved_model, base_model=None):
    # Load and preprocess the image
    dataset = create_inference_dataset(img, batch_size=1)
    # Directly get the image as a numpy array
    preprocessed_img = next(iter(dataset)).numpy()

    # Determine whether to use base_model or saved_model
    model_to_use = base_model if isinstance(
        base_model, tf.keras.Model) else saved_model

    # The target layer (typically the last convolutional layer)
    layer = model_to_use.get_layer(layer_name)

    # Create a Grad-CAM model
    grad_model = tf.keras.models.Model(
        inputs=model_to_use.inputs,
        outputs=[layer.output, model_to_use.output]
    )

    # Predict the class of the input image using saved_model
    preds = saved_model.predict(preprocessed_img)
    class_idx = np.argmax(preds[0])

    # Compute Grad-CAM
    cam = compute_gradcam(preprocessed_img, grad_model, class_idx)

    # Overlay Grad-CAM on the image
    original_img = img.copy()
    heatmap = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap on the original image
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    return superimposed_img




# =================================================================================================
# SIDEBAR:
# =================================================================================================
st.sidebar.header("Sélectionner le modèle:")
is_text_model = st.sidebar.checkbox('Utiliser le modèle de texte', value=True)

# -------------------------------------------------------------------------------------------------
# Select Text Model
text_model_name = st.sidebar.selectbox("Sélectionner le modèle de texte",
    ("ExtraTreesClassifier", "LGBMClassifier", "LogisticRegression", "NearestCentroid",
     "RandomForestClassifier", "XGBClassifier", "VotingClassifier"),
)
if text_model_name in ["NearestCentroid", "VotingClassifier"]:
    st.sidebar.warning("Please note this model cannot predict class probabilities.")

text_methode = st.sidebar.selectbox("Sélectionner la méthode",
    ("Text & Structure", "Text", "Structure"),
)

# -------------------------------------------------------------------------------------------------
# Select Image Model
st.sidebar.divider()
is_image_model = st.sidebar.checkbox("Utiliser le modèle d'image", value=True)

dict_image_model = {
    "CNN": "CNN_ckpt_best_acc.keras",
    "MobileNetV2": "MobileNetV2_ckpt_best_acc.keras",
    "ResNet50": "ResNet50_ckpt_best_acc.keras",
    "SqueezeNet": "SqueezeNet_ckpt_best_acc.keras",
    "EfficientNet": "EfficientNetB1_ckpt_best_acc.keras"
}

image_model_name = st.sidebar.selectbox("Sélectionner le modèle d'image",
    ("MobileNetV2", "CNN", "ResNet50", "SqueezeNet", "EfficientNet"),)
image_model_path = f"models/{dict_image_model[image_model_name]}"

# -------------------------------------------------------------------------------------------------
# Select device:
st.sidebar.divider()
device_option = st.sidebar.selectbox("Select device:", ("CPU", "GPU"))
if device_option == "GPU":
    device = "/GPU:0"
else:
    device = "/CPU:0"

# -------------------------------------------------------------------------------------------------
# Load Models:
if is_text_model:
    text_model = load_text_model(text_methode, text_model_name)

if is_image_model:
    image_model = load_image_model(image_model_path, device)


# =================================================================================================
# Results Tabs:
# =================================================================================================
tab1, tab2, tab3 = st.tabs(["Image", "Résultats du modèle de text", "Résultats du modèle d'image"])

# -------------------------------------------------------------------------------------------------
# Tab Inference / Image preview:
with tab1:

    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "tif"])
    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

    col1, col2 = st.columns([1,5])
    with col1:
        predict_button = st.button("Predict", key="red_button", type='primary')

    with col2:
        if is_text_model and predict_button:
            with st.spinner(f"{text_model_name} inference..."):
                t0 = time.time()
                if text_model_name not in ["NearestCentroid", "VotingClassifier"]:
                    result_text = text_model.predict_proba(image)[0]
                else:
                    result_text = text_model.predict(image)
                    default_cat_list = list(default_categories.values())
                    _idx = default_cat_list.index(result_text)
                    result_text = [0 if i!=_idx else 1 for i in range(6)]
                time_infer_text = time.time() - t0
            print("Pred text:", result_text)
            print(text_model.words)
            print(text_model.words_structure)
        else:
            result_text = None

        if is_image_model and predict_button:
            with st.spinner(f"{image_model_name} inference..."):
                t0 = time.time()
                result_image = inference(image_model, [image], return_predictions=True)
                time_infer_image = time.time() - t0
            print("Pred image:", result_image[0])
            layer_name = layers_gradcam[image_model_name]
            base_model = base_models[image_model_name]
            if base_model is not None:
                base_model = image_model.get_layer(base_model)
            image_gradcam = grad_cam(image, layer_name, image_model, base_model)
        else:
            result_image = None

    colA, colB = st.columns(2)
    if uploaded_image is not None:

        with colA:
            st.header("Image")
            h, w = image.shape[:2]
            image_show = cv2.resize(image.copy(), (int(512/h*w), 512))
            st.image(image_show, channels="BGR", caption='Image', use_column_width=True)

        with colB:
            st.header("Prédictions:")
            if result_text is not None:
                st.divider()
                st.write(f"{text_model_name} :")
                pred_text = default_categories[result_text.argmax()]
                value_result_text = result_text[result_text.argmax()] * 100
                _txt_text = f"{pred_text.capitalize()} : {value_result_text:.02f} %"
                if value_result_text > 85.0:
                    st.success(_txt_text)
                elif 60.0 <= value_result_text <= 85.0:
                    st.warning(_txt_text)
                else:
                    st.error(_txt_text)
                st.write(f"Temps d'inférence: {time_infer_text:.02f}s")

            if result_image is not None:
                st.divider()
                st.write(f"{image_model_name} :")
                pred_image = [default_categories[pred] for pred in result_image.argmax(axis = 1)]
                value_result_image = result_image[0][result_image.argmax(axis = 1)[0]] * 100
                _txt_im = f"{pred_image[0].capitalize()} : {value_result_image:.02f} %"
                if value_result_image > 85.0:
                    st.success(_txt_im)
                elif 60.0 <= value_result_image <= 85.0:
                    st.warning(_txt_im)
                else:
                    st.error(_txt_im)
                st.write(f"Temps d'inférence time: {time_infer_image:.02f}s")

    if result_text is not None or result_image is not None:
        st.divider()
        st.header("Détais:")
        st_plot(result_text, result_image, text_model_name, image_model_name)
    else:
        result_image = None
        result_text = None

# -------------------------------------------------------------------------------------------------
# Tab Text model results:
with tab2:
    if result_text is not None:
        st.header(f"{text_model_name} Prédictions:")
        st_plot(result_text, None, text_model_name, None)
        st.divider()
        st.header("Prédictions de Tesseract:")
        if text_model.osd_result['is_rotated']:
            image_rot = image_rotate(image, text_model.osd_result['rotate'], auto_bound=True)
            tesseract_image = draw_tesseract_result(text_model.ocr_result, image_rot)
        else:
            tesseract_image = draw_tesseract_result(text_model.ocr_result, image)
        st.image(tesseract_image, channels="BGR", caption='Image', use_column_width=True)
        st.header("Structure de mots:")
        structure_col_names = ["count", "lexical diversity"]
        sums = dict()
        names_cat = ["passeport", "email", "invoice", "scientific publication", "handwritten", "national identity card"]
        for cat in names_cat:
            for i in [5, 10, 25, 50]:
                structure_col_names.append(f"keyword {cat} [{i}]")

        data_structure = {}
        data_structure['Label'] = names_cat
        data_structure["Somme"] = [np.sum(text_model.words_structure[2:][4*n: 4+4*n]) for n, _ in enumerate(names_cat)]
        data_structure["Écart type"] = [np.std(text_model.words_structure[2:][4*n: 4+4*n]) for n, _ in enumerate(names_cat)]
        data_structure = pd.DataFrame(data_structure)
        st.bar_chart(data_structure, x="Label", stack=False, horizontal=True)
        st.write({structure_col_names[n] : v for n, v in enumerate(text_model.words_structure)})

# -------------------------------------------------------------------------------------------------
# Tab Image model results:
with tab3:
    if result_image is not None:
        st.header(f"{image_model_name} Prédictions:")
        st_plot(None, result_image[0], None, image_model_name)
        st.divider()
        st.header("Gradcam:")
        if is_image_model and result_image is not None:
            image_gradcam = cv2.resize(image_gradcam, (int(512/h*w), 512))
            st.image(image_gradcam, channels="BGR", caption='Image', use_column_width=False)