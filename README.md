# ExtremeFusion-YOLO-LSTM-AIoT

## 🧠 Introduction
**ExtremeFusion** is an advanced smart monitoring system that integrates real-time human posture estimation, heart rate detection, and environmental sound monitoring.  
By combining **YOLOv8**, **LSTM neural networks**, **ESP32**, and sensors like **MX30105** and **MP34DT05**, this project achieves powerful and real-time multi-modal detection and analysis.

## 🚀 Features
- **Real-Time Pose Estimation** using YOLOv8
- **Behavior Prediction** with LSTM Models
- **Heart Rate and Environmental Sound Monitoring** via ESP32 Sensors
- **Wi-Fi Data Transmission** for IoT integration
- **High-Efficiency Logging** and **Error Frame Capture**
- **FPS Monitoring** for Performance Analysis

## 🛠️ Tech Stack
| Technology | Description |
|:---|:---|
| YOLOv8 | Real-time human pose and object detection |
| LSTM | Time-series human behavior prediction |
| ESP32 | IoT device for sensor data acquisition |
| MX30105 | Heart rate and oxygen sensor |
| MP34DT05 | Digital microphone for sound detection |
| Wi-Fi | Real-time data communication |
| PyTorch | Deep learning framework |
| OpenCV | Video processing and visualization |
| Python | Programming language |

## 🖼️ System Architecture
```
[ESP32 + Sensors] → [Serial Communication] → [YOLOv8 Pose Detection] → [LSTM Behavior Prediction]
        ↓                                                ↓
  [Heart Rate / Sound Data]                       [Real-Time Monitoring & Logging]
        ↓                                                ↓
           [Wi-Fi Transmission] ←→ [Cloud / Edge Integration (Optional)]
```

## 📦 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/ExtremeFusion-YOLO-LSTM-AIoT.git
   cd ExtremeFusion-YOLO-LSTM-AIoT
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure the following models are available:
   - `yolov8n-pose.pt` (for pose estimation)
   - `best16.pt` (for object detection)
   - `LSTM_Model3.pth` (for behavior prediction)

4. Connect your ESP32 device with MX30105 and MP34DT05 sensors.

## 🏃 Usage
Run the main program:
```bash
python main.py
```

**Controls:**
- Press `q` to quit the video display.

**Output:**
- Real-time pose and behavior detection
- Heart rate and sound level monitoring
- Logging results to `detection_log.txt`
- Error frames saved automatically if processing fails

## 📈 Performance
- Real-time detection at **~28-30 FPS** (depending on hardware)
- Supports both **CPU** and **GPU (CUDA)** acceleration
- Automatic error recovery and resource management

## 📝 Log Files
- Detection logs are automatically saved to `detection_log.txt`
- Error frames are saved as `error_frame_YYYYMMDD_HHMMSS.jpg` if an exception occurs

## 🤝 Contribution
Feel free to open an issue or pull request if you find bugs or have suggestions!

## 📜 License
This project is licensed under the MIT License.
