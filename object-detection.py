import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

def run():
    st.header("Object Detection")
    model = YOLO("yolov8n.pt")
    object_names = list(model.names.values())

    with st.form("My Image"):
        uploaded_file = st.file_uploader("Upload Video", type=['png'])
        selected_objects = st.multiselect('Objects to detect', object_names, default=['person'])
        st.form_submit_button(label='submit')

    if uploaded_file is not None:
        image_binary = uploaded_file.read()
        image = np.frombuffer(image_binary, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        results = model.predict(image)
        st.write(results)

if __name__ == "__main__":
    run()

