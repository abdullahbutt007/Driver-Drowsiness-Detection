# 🧠 Real-Time Driver Drowsiness Detection System

## 📌 Overview
This project presents a real-time driver drowsiness detection system developed using machine learning and computer vision techniques. The system analyses facial behaviour from live video input to identify early signs of fatigue and classify the driver as **Alert** or **Drowsy**.

The solution is designed to operate with low latency on a standard laptop, making it suitable for real-time applications without requiring specialised hardware.

---

## 🎯 Objectives
- Detect driver drowsiness in real time using visual cues  
- Minimise false positives through temporal analysis  
- Achieve high classification performance with efficient models  
- Ensure the system runs smoothly on CPU-based environments  

---

## 🛠 Methodology

### Data Sources
- YawDD (Yawning Detection Dataset)  
- SUST-DDD (Driver Drowsiness Dataset)  
- Custom recorded data for real-world testing  

### Feature Extraction
Facial landmarks were extracted using MediaPipe, and the following features were computed:
- **Eye Aspect Ratio (EAR)** → detects eye closure duration  
- **Mouth Aspect Ratio (MAR)** → detects yawning  
- **Head Pose (Pitch, Yaw, Roll)** → detects fatigue-related head movement  

### Model Development
Multiple models were evaluated:
- Support Vector Machine (SVM)  
- Random Forest  
- XGBoost (selected final model)

The final model was trained using scaled features and optimised for real-time inference.

---

## 📊 Results & Performance
- **Accuracy:** ~88%  
- **F1 Score:** ~0.88  
- **ROC-AUC:** ~0.91  
- **Inference Time:** <10 ms per frame (CPU)  

### Key Observations
- XGBoost provided the best balance between accuracy and speed  
- Temporal logic (e.g. sustained eye closure) significantly reduced false positives  
- Combining EAR, MAR, and head pose improved robustness compared to single-feature systems  

---

## ⚙️ System Architecture
1. Capture real-time video input via webcam  
2. Detect face and extract landmarks using MediaPipe  
3. Compute EAR, MAR, and head pose features  
4. Apply feature scaling (Power Transformer)  
5. Predict driver state using trained XGBoost model  
6. Display output: **Alert / Drowsy** (with optional alert trigger)  

---

## 🚀 Features
- Real-time detection using webcam input  
- Multi-feature fatigue detection (eyes, mouth, head)  
- Machine learning-based classification (not rule-based only)  
- Lightweight and efficient (runs on MacBook CPU)  
- Modular and extendable system design  

---

## 🔮 Future Improvements
- Improve performance in low-light and night-time conditions  
- Expand dataset diversity (lighting, demographics, camera angles)  
- Deploy as a mobile or embedded system  
- Integrate audio/visual alert mechanisms  
- Explore deep learning-based temporal models (e.g. LSTM, CNN-LSTM)  

---

## 📫 Contact
If you'd like to discuss this project or collaborate, feel free to connect with me on LinkedIn.

---

## ▶️ How to Run

1. Clone the repository  
2. Install dependencies:
   pip install -r requirements.txt  

3. Run the script:
   python main.py  
