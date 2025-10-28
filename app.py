import os
import csv
import datetime
import threading
import queue
import pathlib
import sqlite3
import cv2
import torch
import tkinter as tk
from tkinter import filedialog, messagebox

try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False

try:
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# ---------------- Database Setup ----------------
DB_FILE = "hazards.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS hazard_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hazard_name TEXT,
            confidence REAL,
            timestamp TEXT,
            source TEXT
        )
    """)
    conn.commit()
    conn.close()

def store_in_db(hazard_name, confidence, source):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO hazard_log (hazard_name, confidence, timestamp, source) VALUES (?, ?, ?, ?)",
              (hazard_name, confidence, datetime.datetime.now().isoformat(sep=' ', timespec='seconds'), str(source)))
    conn.commit()
    conn.close()

# Initialize DB once
init_db()

# ---------------- Constants ----------------
HAZARD_CLASSES = ['person', 'car', 'truck', 'bicycle', 'motorbike', 'pothole']
COLORS = {
    'person': (0, 255, 0),
    'car': (255, 0, 0),
    'truck': (0, 0, 255),
    'bicycle': (255, 255, 0),
    'motorbike': (0, 255, 255),
    'pothole': (128, 0, 128)
}
SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

detection_thread = None
stop_event = threading.Event()
status_queue = queue.Queue()
tts_queue = queue.Queue()

# ---------------- TTS Thread ----------------
def tts_worker():
    while True:
        msg = tts_queue.get()
        if msg is None:
            break
        try:
            if TTS_AVAILABLE:
                engine.say(msg)
                engine.runAndWait()
            elif PYGAME_AVAILABLE and os.path.exists("alert.mp3"):
                pygame.mixer.music.load("alert.mp3")
                pygame.mixer.music.play()
        except Exception:
            pass
        tts_queue.task_done()

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

# ---------------- Load YOLOv5 ----------------
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.conf = 0.4
except Exception:
    model = None

# ---------------- Detection Function ----------------
def run_detection(source):
    if model is None:
        status_queue.put("Model not available.")
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        status_queue.put("❌ Cannot open video source!")
        return

    frame_count = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            status_queue.put("Video ended or cannot read frame.")
            break

        frame_count += 1
        results = model(frame)
        detections = results.pandas().xyxy[0]
        hazard_count = {cls: 0 for cls in HAZARD_CLASSES}
        hazard_detected_any = False
        last_detected_name = None

        for _, row in detections.iterrows():
            class_name = str(row['name'])
            confidence = float(row['confidence'])
            if class_name in HAZARD_CLASSES:
                hazard_detected_any = True
                hazard_count[class_name] += 1
                last_detected_name = class_name
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                color = COLORS.get(class_name, (0, 0, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(frame, label, (x1, max(y1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Log in database
                store_in_db(class_name, confidence, source)

        # Display on screen
        cv2.putText(frame, f"Frame: {frame_count}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        y_offset = 45
        for cls, cnt in hazard_count.items():
            cv2.putText(frame, f"{cls}: {cnt}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            y_offset += 22

        if hazard_detected_any:
            cv2.putText(frame, "⚠️ HAZARD DETECTED", (frame.shape[1]-330, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            if last_detected_name:
                tts_queue.put(f"Warning! {last_detected_name} ahead!")
        else:
            cv2.putText(frame, "✅ No hazards", (frame.shape[1]-200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Road Hazard Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_event.set()
            break
        if key == ord('s'):
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(SNAPSHOT_DIR, f"snapshot_{ts}.jpg")
            cv2.imwrite(filename, frame)
            status_queue.put(f"Snapshot saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    status_queue.put("Stopped")

# ---------------- GUI ----------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Road Hazard Detection with Database")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        tk.Label(root, text="YOLOv5 Road Hazard Detection", font=("Arial", 16, "bold")).pack(pady=8)
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=6)

        self.webcam_btn = tk.Button(btn_frame, text="Use Webcam", command=self.start_webcam, width=18, height=2)
        self.webcam_btn.grid(row=0, column=0, padx=6)
        self.video_btn = tk.Button(btn_frame, text="Select Video File", command=self.select_video, width=18, height=2)
        self.video_btn.grid(row=0, column=1, padx=6)
        self.stop_btn = tk.Button(btn_frame, text="Stop", command=self.stop_detection, width=18, height=2, state="disabled")
        self.stop_btn.grid(row=0, column=2, padx=6)

        self.status_var = tk.StringVar(value="Status: Idle")
        tk.Label(root, textvariable=self.status_var, font=("Arial", 12)).pack(pady=6)
        tk.Label(root, text="Press 'q' to quit video, 's' to save snapshot", font=("Arial", 9, "italic")).pack(pady=4)
        self.root.after(300, self.poll_status_queue)

    def start_webcam(self):
        global detection_thread
        if detection_thread and detection_thread.is_alive():
            messagebox.showinfo("Already running", "Detection is already running.")
            return
        stop_event.clear()
        self.stop_btn.config(state="normal")
        self.webcam_btn.config(state="disabled")
        self.video_btn.config(state="disabled")
        detection_thread = threading.Thread(target=run_detection, args=(0,), daemon=True)
        detection_thread.start()
        self.status_var.set("Status: Starting webcam...")

    def select_video(self):
        global detection_thread
        filename = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if not filename:
            return
        if detection_thread and detection_thread.is_alive():
            messagebox.showinfo("Already running", "Stop current detection before starting a new one.")
            return
        stop_event.clear()
        self.stop_btn.config(state="normal")
        self.webcam_btn.config(state="disabled")
        self.video_btn.config(state="disabled")
        detection_thread = threading.Thread(target=run_detection, args=(filename,), daemon=True)
        detection_thread.start()
        self.status_var.set(f"Status: Playing {pathlib.Path(filename).name}")

    def stop_detection(self):
        stop_event.set()
        self.stop_btn.config(state="disabled")
        self.webcam_btn.config(state="normal")
        self.video_btn.config(state="normal")
        self.status_var.set("Status: Stopping...")

    def poll_status_queue(self):
        try:
            while not status_queue.empty():
                msg = status_queue.get_nowait()
                self.status_var.set(f"Status: {msg}")
                status_queue.task_done()
                if msg in ("Stopped", "Video ended or cannot read frame.", "❌ Cannot open video source!", "Model not available."):
                    self.stop_btn.config(state="disabled")
                    self.webcam_btn.config(state="normal")
                    self.video_btn.config(state="normal")
        except Exception:
            pass
        self.root.after(300, self.poll_status_queue)

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you really want to quit?"):
            stop_event.set()
            try:
                tts_queue.put(None)
            except Exception:
                pass
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

