import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

def run():
    st.header("Object Detection")
    model = YOLO("yolov8n.pt")
    object_names = list(model.names.values())

    with st.form("My Image"):
        uploaded_file = st.file_uploader("Upload Image", type=['png'])
        st.form_submit_button(label='Detect')

    if uploaded_file is not None:
        image_binary = uploaded_file.read()
        image = np.frombuffer(image_binary, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        results = model.predict(image)[0]
        image = np.copy(image[:,:,::-1])
        for data in results.boxes.data.tolist():
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            class_id = int(data[5])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.rectangle(image, (xmin, ymin - 20), (xmin, ymin + 20), (0, 255, 0), -1)
            cv2.putText(image, str(results.names[class_id]), (xmin + 5, ymin + 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
        fig, ax = plt.subplots()
        ax.imshow(image)
        st.pyplot(fig)

if __name__ == "__main__":
    run()

