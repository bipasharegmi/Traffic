import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import joblib
from ultralytics import YOLO

# Load model and scaler
model = joblib.load("rbf_model.pkl")
scaler = joblib.load("scaler.pkl")
yolo = YOLO("yolov8n.pt")

VEHICLE_CLASSES = [2, 3, 5, 7]  # Car, Motorcycle, Bus, Truck

# Function to process video and predict congestion
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video.")
        return

    frame_rate = int(cap.get(cv2.CAP_PROP_FPS) or 30)
    road_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)

    prev_positions = {}
    vehicle_count_sum = 0
    speed_data = []
    total_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1

        results = yolo(frame)
        new_positions = {}
        vehicle_count = 0

        for result in results:
            if result.boxes is None:
                continue

            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                x1, y1, x2, y2 = box
                class_id = int(cls)

                if class_id in VEHICLE_CLASSES:
                    vehicle_count += 1
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    new_positions[class_id] = (center_x, center_y)

        vehicle_count_sum += vehicle_count

        for class_id, (x, y) in new_positions.items():
            if class_id in prev_positions:
                distance = np.sqrt((x - prev_positions[class_id][0]) ** 2 + (y - prev_positions[class_id][1]) ** 2)
                speed = (distance * frame_rate) / 10
                speed_data.append(speed)

        prev_positions = new_positions

    cap.release()

    avg_vehicle_count = vehicle_count_sum / total_frames if total_frames else 0
    avg_speed = np.mean(speed_data) if speed_data else 0

    features = np.array([[avg_vehicle_count, avg_speed]])
    features_scaled = scaler.transform(features)
    predicted_congestion = model.predict(features_scaled)[0]
    return predicted_congestion

# GUI function to upload video
def upload_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mov *.avi")])
    if file_path:
        result = process_video(file_path)
        result_label.config(text=f"Predicted Congestion Level: {result:.2f}")

# Tkinter GUI setup
root = tk.Tk()
root.title("🚦 Traffic Congestion Predictor")
root.geometry("500x300")
root.configure(bg="#eaf6f6")

# Optional: set app icon
# root.iconbitmap("icon.ico")  # Add a .ico icon if you have one

# Header
header_frame = tk.Frame(root, bg="#00bcd4")
header_frame.pack(fill="x")

header_label = tk.Label(header_frame, text=" Big data Traffic Congestion Predictor",
                        font=("Helvetica", 16, "bold"), fg="white", bg="#00bcd4", pady=10)
header_label.pack()

# Content Frame
content_frame = tk.Frame(root, bg="#eaf6f6", padx=20, pady=20)
content_frame.pack(expand=True)

# Upload instruction
instruction_label = tk.Label(content_frame, text="Upload a traffic video to analyze congestion level",
                             font=("Arial", 12), bg="#eaf6f6")
instruction_label.pack(pady=10)

# Upload Button
upload_button = tk.Button(content_frame, text="📁 Upload Video & Predict",
                          font=("Arial", 12, "bold"), bg="#00796B", fg="white",
                          padx=15, pady=8, command=upload_video, relief="raised", bd=3)
upload_button.pack(pady=15)

# Result label
result_label = tk.Label(content_frame, text="", font=("Arial", 12, "bold"),
                        bg="#eaf6f6", fg="#004d40")
result_label.pack(pady=10)

# Footer
footer_label = tk.Label(root, text=" Smart City Project",
                        font=("Arial", 9), bg="#eaf6f6", fg="#555")
footer_label.pack(side="bottom", pady=5)

root.mainloop()
