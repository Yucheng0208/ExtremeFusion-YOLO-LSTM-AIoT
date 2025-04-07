import serial
import time
import threading
import queue
from contextlib import closing
from ultralytics import YOLO
import torch
import numpy as np
import cv2
import logging
import gc
import datetime
from lstm import LSTM_Model

# Hide YOLO outputs
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

# Configuration
SERIAL_PORT = 'COM3'
BAUD_RATE = 115200
VIDEO_PATH = "0"
POSE_MODEL_PATH = 'yolov8n-pose.pt'
OBJECT_MODEL_PATH = 'best.pt'
LSTM_MODEL_PATH = "LSTM_Model.pth"
FRAME_INTERVAL = 300
CONFIDENCE_THRESHOLD = 0.7

# Shared variables
heart_rate = 0
volume = 0
output_queue = queue.Queue()
stop_event = threading.Event()

# Class color settings
def get_color_for_class(cls_index):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    return colors[cls_index % len(colors)]

# Save detection log
def save_log(content):
    with open('detection_log.txt', 'a', encoding='utf-8') as f:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"[{timestamp}] {content}\n")

# Terminal output thread
def print_status():
    while not stop_event.is_set():
        try:
            data = output_queue.get(timeout=1)
            if data is None:
                break
            print(data)
        except queue.Empty:
            continue

# Serial reading thread
def read_serial_data(ser):
    global heart_rate, volume
    while not stop_event.is_set():
        if ser.in_waiting > 0:
            try:
                data = ser.readline().decode('utf-8').strip()
                parts = data.split(',')
                if len(parts) == 2:
                    heart_rate = float(parts[0].strip())
                    volume = int(parts[1].strip())
                else:
                    output_queue.put(f"Error: Incorrect data format received: {data}")
            except ValueError:
                output_queue.put("Error: Invalid number format")

# Frame processing function
def process_frame(frame, pose_model, object_model, lstm_model):
    normal = rest = fall = 0
    object_detected = []

    pose_results = pose_model.predict(frame)[0]
    object_results = object_model(frame, conf=CONFIDENCE_THRESHOLD)

    if pose_results.keypoints.conf is not None:
        keypoints = pose_results.keypoints.xyn.tolist()
        confs = pose_results.boxes.conf.tolist()
        keypointsconf = pose_results.keypoints.conf.tolist()

        npconfs = np.array(confs)
        npkeypoints = np.array(keypoints)
        npkeypointsconf = np.array(keypointsconf)

        for idx in range(npkeypoints.shape[0]):
            if npconfs[idx] >= 0.5:
                data_listx, data_listy = [], []
                for k in range(17):
                    if npkeypointsconf[idx][k] >= 0.5:
                        data_listx.append(npkeypoints[idx][k][0])
                        data_listy.append(npkeypoints[idx][k][1])
                    else:
                        data_listx.append(-1)
                        data_listy.append(-1)

                data_test = np.vstack([data_listx, data_listy])
                data_test = np.reshape(data_test, (-1, 1, 34))

                lstm_input = torch.tensor(data_test, dtype=torch.float32)
                if torch.cuda.is_available():
                    lstm_input = lstm_input.cuda()

                outputs = lstm_model(lstm_input)
                outputs = outputs.cpu()
                _, predicted = torch.max(outputs, 1)

                if np.count_nonzero(data_test == -1) <= 34:
                    if predicted.item() == 0:
                        normal += 1
                    elif predicted.item() == 1:
                        rest += 1
                    elif predicted.item() == 2:
                        fall += 1

        detection_summary = f"Pose Detection - Total: {npkeypoints.shape[0]}, Normal: {normal}, Resting: {rest}, Abnormal: {fall}"
        output_queue.put(f"### {detection_summary}")
        save_log(detection_summary)

    if object_results:
        for result in object_results[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = result[:6]
            if conf >= CONFIDENCE_THRESHOLD:
                label = f"{object_model.names[int(cls)]} {conf:.2f}"
                color = get_color_for_class(int(cls))
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                frame = cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                object_detected.append(f"Detected {object_model.names[int(cls)]}, Confidence: {conf:.2f}")

    for item in object_detected:
        output_queue.put(item)

    annotated_frame = pose_results.plot()
    small_frame = cv2.resize(annotated_frame, (720, 480))
    cv2.imshow("YOLOv8 Pose Estimation and Object Detection", small_frame)

    output_queue.put("@@@ Physiological Detection")
    output_queue.put(f"Heart Rate: {heart_rate}, Volume: {volume}")
    gc.collect()

# Main program
if __name__ == "__main__":
    with closing(serial.Serial(SERIAL_PORT, BAUD_RATE)) as ser, closing(cv2.VideoCapture(VIDEO_PATH)) as cap:
        pose_model = YOLO(POSE_MODEL_PATH)
        object_model = YOLO(OBJECT_MODEL_PATH)
        lstm_model = LSTM_Model()
        lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location='cpu'))
        if torch.cuda.is_available():
            lstm_model = lstm_model.cuda()
        lstm_model.eval()

        threading.Thread(target=print_status, daemon=True).start()
        threading.Thread(target=read_serial_data, args=(ser,), daemon=True).start()

        frame_count = 0
        counter = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Video playback finished, exiting program...")
                break

            frame_count += 1

            if frame_count % FRAME_INTERVAL == 0:
                counter += 1
                print(f"\nDetection Count ===> {counter}")
                try:
                    process_frame(frame, pose_model, object_model, lstm_model)

                    elapsed_time = time.time() - start_time
                    fps = FRAME_INTERVAL / elapsed_time
                    print(f"FPS: {fps:.2f}")
                    start_time = time.time()

                except Exception as e:
                    error_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    error_filename = f'error_frame_{error_time}.jpg'
                    cv2.imwrite(error_filename, frame)
                    print(f"Error during processing: {e}")
                    print(f"Saved error frame as {error_filename}")
                    continue

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        stop_event.set()
        output_queue.put(None)
        time.sleep(1)

    cv2.destroyAllWindows()
