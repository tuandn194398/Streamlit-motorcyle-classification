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
st.title("Interface for Bộ Công An")

# sidebar
st.sidebar.header("DL Model Config")

# model options
task_type = st.sidebar.selectbox(
    "Select Task",
    ["Detection",
    "Classify"]
)

model_type = None
if task_type == "Detection":
    model_type = st.sidebar.selectbox(
        "Select Model",
        config.DETECTION_MODEL_LIST
    )
elif task_type == "Classify":
    model_type = st.sidebar.selectbox(
        "Select Model",
        config.CLASSIFY_MODEL_LIST
    )
else:
    st.error("Currently only 'Detection' function is implemented")

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 30, 100, 50)) / 100

model_path = ""
if model_type:
    model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
else:
    st.error("Please Select Model in Sidebar")

# load pretrained DL model
try:
    model = load_Yolo_model(model_path)
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {model_path}")

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
    infer_uploaded_image(confidence, model)
elif source_selectbox == config.SOURCES_LIST[1]: # Video
    infer_uploaded_video(confidence, model)
elif source_selectbox == config.SOURCES_LIST[2]: # Webcam
    webcam_url = st.sidebar.text_input("Enter Webcam URL")
    if webcam_url:
        infer_uploaded_webcam(confidence, model, webcam_url)
else:
    st.error("Currently only 'Image' and 'Video' source are implemented")

# Check if the "Analyze" button has been pressed
if 'analyze_pressed' in st.session_state and st.session_state['analyze_pressed']:
    # Display the select box for classification models
    classify_model = st.selectbox(
        "Select Classification Model",
        config.CLASSIFY_MODEL_LIST
    )
    if model_type:
        model_path = Path(config.CLASSIFY_MODEL_DIR, str(classify_model))
    else:
        "select model"
    print(model_path)
    # Reset the session state so the select box won't keep appearing
    st.session_state['analyze_pressed'] = False
