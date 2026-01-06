import streamlit as st
import cv2
import face_recognition
import numpy as np
from utils import Database

st.set_page_config(page_title="Register User")
st.title("Register New User")
db = Database()

col1, col2 = st.columns(2)
new_id = col1.text_input("Employee ID")
new_name = col2.text_input("Full Name")

if st.button("Start Capture", type="primary"):
    if not new_id or not new_name:
        st.error("Fill all fields.")
    else:
        cap = cv2.VideoCapture(0)
        try:
            img_place = st.empty()
            bar = st.progress(0)
            samples = []
            
            while len(samples) < 20:
                ret, frame = cap.read()
                if not ret: break
                
                small = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                boxes = face_recognition.face_locations(rgb)
                
                if len(boxes) == 1:
                    samples.append(face_recognition.face_encodings(rgb, boxes)[0])
                    top, right, bottom, left = [int(v/0.25) for v in boxes[0]]
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2, lineType=cv2.LINE_AA)
                
                bar.progress(len(samples)/20)
                img_place.image(frame, channels="BGR", use_container_width=True)

            if len(samples) == 20:
                avg_enc = np.mean(samples, axis=0)
                success, msg = db.add_user(new_id, new_name, avg_enc)
                if success:
                    st.balloons()
                    st.success(f"Registered {new_name}!")
                    if 'known_users' in st.session_state: del st.session_state['known_users']
                else: st.error(msg)
        finally:
            # RELEASE CAMERA HARDWARE
            cap.release()
            cv2.destroyAllWindows()