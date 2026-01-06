import sqlite3
import numpy as np
import mediapipe as mp
from datetime import datetime
import cv2

# --- CONFIGURATION ---
DB_FILE = "attendance_system.db"
COOLDOWN_SECONDS = 100  
MATCH_THRESHOLD = 0.55
EAR_THRESHOLD = 0.26
CONSECUTIVE_FRAMES = 1

def play_success_sound():
    try:
        import winsound
        winsound.Beep(1000, 200)
    except:
        pass

class Database:
    def __init__(self):
        self.conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    emp_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    encoding BLOB NOT NULL
                )
            ''')
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    emp_id TEXT,
                    timestamp DATETIME,
                    FOREIGN KEY(emp_id) REFERENCES users(emp_id)
                )
            ''')

    def add_user(self, emp_id, name, encoding):
        try:
            binary_encoding = encoding.tobytes()
            with self.conn:
                self.conn.execute("INSERT INTO users (emp_id, name, encoding) VALUES (?, ?, ?)", 
                                  (emp_id, name, binary_encoding))
            return True, "Success"
        except sqlite3.IntegrityError:
            return False, "ID already exists!"

    def get_all_users(self):
        cursor = self.conn.execute("SELECT emp_id, name, encoding FROM users")
        users = []
        for row in cursor:
            users.append({
                "id": row[0], 
                "name": row[1], 
                "enc": np.frombuffer(row[2], dtype=np.float64)
            })
        return users

    def delete_user(self, emp_id):
        try:
            with self.conn:
                self.conn.execute("DELETE FROM logs WHERE emp_id = ?", (emp_id,))
                self.conn.execute("DELETE FROM users WHERE emp_id = ?", (emp_id,))
            return True
        except Exception:
            return False

    def is_in_cooldown(self, emp_id):
        cursor = self.conn.execute(
            "SELECT timestamp FROM logs WHERE emp_id = ? ORDER BY timestamp DESC LIMIT 1", 
            (emp_id,)
        )
        row = cursor.fetchone()
        if row:
            last_time = datetime.fromisoformat(row[0])
            elapsed = (datetime.now() - last_time).total_seconds()
            return elapsed < COOLDOWN_SECONDS
        return False

    def mark_attendance(self, emp_id):
        if self.is_in_cooldown(emp_id):
            return False
        with self.conn:
            self.conn.execute("INSERT INTO logs (emp_id, timestamp) VALUES (?, ?)", 
                              (emp_id, datetime.now().isoformat()))
        return True

class FaceLogic:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
        )
        self.blink_counter = 0

    def check_blink(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb)
        if not results.multi_face_landmarks: return False
        
        lm = results.multi_face_landmarks[0].landmark
        left = [33, 160, 158, 133, 153, 144]
        right = [362, 385, 387, 263, 373, 380]
        
        def eye_ratio(idx):
            p = np.array([[lm[i].x * w, lm[i].y * h] for i in idx])
            v = np.linalg.norm(p[1]-p[5]) + np.linalg.norm(p[2]-p[4])
            h_d = np.linalg.norm(p[0]-p[3]) * 2.0
            return v / h_d

        ear = (eye_ratio(left) + eye_ratio(right)) / 2.0
        if ear < EAR_THRESHOLD:
            self.blink_counter += 1
            return False
        else:
            if self.blink_counter >= CONSECUTIVE_FRAMES:
                self.blink_counter = 0
                return True
            self.blink_counter = 0
            return False