from ultralytics import YOLO
import streamlit as st
import cv2

def _display_detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    # Resize the image to a standard size
    image = cv2.resize(image,  (720, int(720 * (9 / 16))))

    # Predict the objects in the image using YOLOv8 model
    res = model.predict(image, conf=conf, classes=3, save_crop = True, save_txt = True)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


def _display_classify_frame(conf, model, st_frame, image):
    res = model.predict(image, conf= conf, show_labels = True)
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected image',
                   channels="BGR",
                   use_column_width=True
                   )