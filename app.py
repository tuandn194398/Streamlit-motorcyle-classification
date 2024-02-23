#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
from PIL import Image
import streamlit as st
import config
# from utils import load_Yolo_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam
from func.copy_file import copy_folder

from ultralytics import YOLO
import streamlit as st
import cv2
import fnmatch
import os
from PIL import Image
import tempfile
import time
from  func.delete_file import remove_folder_contents
from func.mode_infer import _display_detected_frames, _display_classify_frame
from func.csv_generator import csv_generate
# Global variable to control the video playback
video_playing = True

def stop_video():
    global video_playing
    video_playing = False

@st.cache_resource
def load_Yolo_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model

# def load_model(model_path):
#     return 1

def infer_uploaded_image(conf, model):
    """
    Execute inference for uploaded image
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # adding the uploaded image to the page with caption
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_column_width=True
            )

    if source_img:
        if st.button("Detect"):
            remove_folder_contents('runs/detect')
            with st.spinner("Running..."):
                res = model.predict(uploaded_image,
                                    conf=conf, classes=3)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]

                with col2:
                    st.image(res_plotted,
                             caption="Detected Image",
                             use_column_width=True)
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.xywh)
                    except Exception as ex:
                        st.write("No image is uploaded yet!")
                        st.write(ex)


def infer_uploaded_video(conf, detect_model, classify_model):
    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    global video_playing
    video_placeholder = st.empty()
    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )

    if source_video:
        video_placeholder.video(source_video)
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Detect motorbike"):
                if remove_folder_contents('runs/detect') and remove_folder_contents('runs/classify'):
                    video_playing = True
                    bbox_images_folder = './runs/detect/predict/crops/motorcycle'
                    with st.spinner("Running..."):
                        try:
                            tfile = tempfile.NamedTemporaryFile(delete=False)
                            tfile.write(source_video.read())
                            vid_cap = cv2.VideoCapture(tfile.name)
                            # st_frame = st.empty()
                            while video_playing and vid_cap.isOpened():
                                success, image = vid_cap.read()
                                if success:
                                    _display_detected_frames(conf, detect_model, video_placeholder, image)
                                else:
                                    vid_cap.release()
                                    break
                        except Exception as e:
                            st.error(f"Error loading video: {e}")
                        
                        try:
                            # Load your model here (assuming 'load_model' is a function to do so)
                            # List all image files in the directory
                            for file in os.listdir(bbox_images_folder): 
                                if fnmatch.fnmatch(file, '*.jpg'):  # Check if the file is a .jpg
                                    path = os.path.join(bbox_images_folder, file)
                                    # Predict and display each image
                                    if path and os.path.isfile(path):
                                        classify_model.predict(path, save_txt=True)
                                        # Assuming 'results' has a property that contains the image data
                                        # st_frame.image(results.img_data, use_column_width=True)
                                        time.sleep(0.1)
                            csv_generate()
                        except Exception as e:
                            st.error(f"An error occurred: {e}")
        with col2:
            if st.button("Stop Video"):
                stop_video()
                        
                                        

def infer_uploaded_webcam(conf, model, webcam_url):
    """
    Execute inference for webcam.
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    global video_playing
    try:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Webcam"):
                video_playing = True
                remove_folder_contents('runs/detect')
        with col2:
            if st.button("Stop Webcam"):
                stop_video()
        vid_cap = cv2.VideoCapture(webcam_url)  # local camera
        st_frame = st.empty()
        while video_playing:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(conf, model, st_frame, image)
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")

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
