import streamlit as st
import cv2
import face_recognition
import numpy as np
import threading
from utils import Database, FaceLogic, play_success_sound, MATCH_THRESHOLD

# --- CONSTANTS ---
TARGET_WIDTH = 1500
TARGET_HEIGHT = 750
RECOGNITION_INTERVAL = 10 
SCALE_FACTOR = 0.25  

st.set_page_config(page_title="Attendance System", layout="wide")

class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if grabbed:
                self.frame = frame

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        if self.stream.isOpened():
            self.stream.release()

# --- APP START ---
st.title("Attendance System")
db = Database()
logic = FaceLogic()
if 'known_users' not in st.session_state:
    st.session_state['known_users'] = db.get_all_users()
    
users = st.session_state['known_users']
run_system = st.checkbox("Turn On Camera")
placeholder = st.empty()

if run_system:
    stream = VideoStream(0).start()

    try:
        count = 0
        current_data = [] 

        while run_system:
            frame = stream.read()
            if frame is None: break
            
            is_blink = logic.check_blink(frame)

            # --- RECOGNITION BLOCK ---
            if count % RECOGNITION_INTERVAL == 0:
                small = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                
                boxes = face_recognition.face_locations(rgb)
                encodings = face_recognition.face_encodings(rgb, boxes)
                
                current_data = []
                for box, enc in zip(boxes, encodings):
                    # Default state for unidentified faces
                    display_text = "Unknown"
                    color = (0, 0, 255) # Red for unknown
                    is_known = False
                    
                    if users:
                        dists = face_recognition.face_distance([u['enc'] for u in users], enc)
                        best = np.argmin(dists)
                        if dists[best] < MATCH_THRESHOLD:
                            is_known = True
                            name = users[best]['name']
                            eid = users[best]['id']
                            
                            # If known, show Name + Status
                            if db.is_in_cooldown(eid):
                                display_text = f"{name}: Marked"
                                color = (0, 255, 0)
                            else:
                                display_text = f"{name}: Blink!"
                                color = (0, 165, 255)
                    
                    scaled_box = [int(v / SCALE_FACTOR) for v in box]
                    current_data.append({
                        "box": scaled_box, 
                        "text": display_text, 
                        "color": color,
                        "is_known": is_known,
                        "name": name if is_known else None
                    })

            # --- ACTION ON BLINK ---
            if is_blink:
                for person in current_data:
                    if person['is_known'] and "Blink!" in person['text']:
                        # Get ID from name
                        emp_id = next(u['id'] for u in users if u['name'] == person['name'])
                        if db.mark_attendance(emp_id):
                            play_success_sound()
                            person['text'] = f"{person['name']}: Success!"
                            person['color'] = (0, 255, 0)

            # --- DRAWING ---
            for p in current_data:
                top, right, bottom, left = p['box']
                cv2.rectangle(frame, (left, top), (right, bottom), p['color'], 2, cv2.LINE_AA)
                cv2.putText(frame, p['text'], (left, top-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, p['color'], 2, cv2.LINE_AA)

            # --- FINAL RESIZE (1000x500) ---
            h, w = frame.shape[:2]
            crop_h = int(w / 2) 
            start_y = max(0, (h - crop_h) // 2)
            cropped = frame[start_y : start_y + crop_h, 0 : w]
            
            final = cv2.resize(cropped, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
            placeholder.image(final, channels="BGR")
            count += 1
            
    finally:
        stream.stop()