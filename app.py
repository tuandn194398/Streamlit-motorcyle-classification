#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
from PIL import Image
import streamlit as st
import config
from utils import load_Yolo_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam
from func.copy_file import copy_folder
# setting page layout
st.set_page_config(
    page_title="Interactive Interface for Motorcycle",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# main page heading
st.title("Interface for TH Truemilk ")

# sidebar
st.sidebar.header("DL Model Config")


# model options
chose_detect_model = st.sidebar.selectbox(
        "Select Detect Model",
        config.DETECTION_MODEL_LIST
)
chose_classify_model = st.sidebar.selectbox(
        "Select Classify Model",
        config.CLASSIFY_MODEL_LIST
)


confidence = float(st.sidebar.slider(
    "Select Model Confidence", 30, 100, 50)) / 100

detect_model_path = ""
classify_model_path = ""
if chose_detect_model and chose_classify_model:
    detect_model_path = Path(config.DETECTION_MODEL_DIR, str(chose_detect_model))
    classify_model_path = Path(config.CLASSIFY_MODEL_DIR, str(chose_classify_model) )
else:
    st.error("You haven't chosen both models yet.")

# load pretrained DL model
try:
    detect_model = load_Yolo_model(detect_model_path)
    classify_model = load_Yolo_model(classify_model_path)
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path:")

#save toggle
save_toggle = st.sidebar.checkbox(
    "Save",
    key = "save_toggle"
)
if save_toggle:
    source_path = "runs/detect"  # The source folder you want to copy
    destination_path = "./temp/"  # The destination folder
    copy_folder(source_path, destination_path)
    # Reset the checkbox state after the operation
    save_toggle = False

# image/video options
st.sidebar.header("Image/Video Config")
source_selectbox = st.sidebar.selectbox(
    "Select Source",
    config.SOURCES_LIST
)

source_img = None
if source_selectbox == config.SOURCES_LIST[0]: # Image
    infer_uploaded_image(confidence, detect_model)
elif source_selectbox == config.SOURCES_LIST[1]: # Video
    infer_uploaded_video(confidence, detect_model, classify_model)
elif source_selectbox == config.SOURCES_LIST[2]: # Webcam
    webcam_url = st.sidebar.text_input("Enter Webcam URL")
    if webcam_url:
        infer_uploaded_webcam(confidence, detect_model, webcam_url)
else:
    st.error("Currently only 'Image' and 'Video' source are implemented")

# Check if the "Analyze" button has been pressed
if 'analyze_pressed' in st.session_state and st.session_state['analyze_pressed']:
    # Display the select box for classification models
    classify_model = st.selectbox(
        "Select Classification Model",
        config.CLASSIFY_MODEL_LIST
    )
    try:
        model = classify_model
    except Exception as e:
        st.error(f"Unable to load model. Please check the specified path: {classify_model_path}")
    # Reset the session state so the select box won't keep appearing
    st.session_state['analyze_pressed'] = False
