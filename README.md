# 🤟 Sign Language to Speech Conversion System  

**MediaPipe · ANN · Groq LLM · ElevenLabs · Flask · Docker · AWS CI/CD**

A full-stack **Sign Language to Speech** web application that detects hand gestures from images, videos, or live webcam input, converts them into text using **hand-landmark–based ANN classification**, refines sentences using **Groq LLM**, and finally converts text into **natural human-like speech using ElevenLabs**.

> 🚀 Built for **accuracy, scalability, and production deployment**.

---

## ✨ Key Features

- ✋ **MediaPipe hand landmark extraction** (supports **2 hands**)
- 🧠 **ANN-based gesture classification** (landmark-driven, not raw images)
- 📐 **Normalized & scale-invariant hand pose features**
- 🧠 **Groq LLM** for sentence and grammar refinement
- 🔊 **ElevenLabs Text-to-Speech**
- 🌐 **Flask web application**
- 🐳 **Dockerized deployment**
- ☁️ **AWS EC2 + ECR**
- 🔁 **GitHub Actions CI/CD**
- 🔐 Secure API key handling via environment variables

---

## 🧠 System Architecture

Image / Video / Webcam
↓
MediaPipe Hands
(21 landmarks × 2 hands)
↓
Landmark Normalization

Wrist-relative

Unit scaling

Zero padding
↓
ANN Classifier (126 features)
↓
Raw Text Prediction
↓
Groq LLM (NLP Refinement)
↓
Final Sentence
↓
ElevenLabs TTS
↓
Speech Output 🔊


---

## ✋ Landmark Extraction Pipeline

- Detect up to **2 hands per frame**
- Extract **21 landmarks per hand** `(x, y, z)`
- Normalize landmarks:
  - Relative to wrist
  - Scale-invariant normalization
- If one hand is missing → **zero-pad**
- Final feature vector size:

2 × 21 × 3 = 126 features


Saved as `.npy` files per label.

---

## 🧠 ANN Model Architecture

Input (126)
↓
Dense(256) + BatchNorm + Dropout(0.3)
↓
Dense(128) + BatchNorm + Dropout(0.25)
↓
Dense(64)
↓
Dense(N classes) + Softmax


### 🔧 Training Configuration

| Parameter | Value |
|--------|------|
Epochs | 80 |
Batch Size | 32 |
Optimizer | Adam |
Learning Rate | 1e-3 |
Loss | Categorical Crossentropy |
Validation Split | 20% |
Callbacks | EarlyStopping, ReduceLROnPlateau |

---

## 🧠 NLP with Groq

- Refines:
  - Broken words
  - Partial predictions
  - Contextual meaning
- Converts raw gesture outputs into **human-readable sentences**
- Applied **after classification**, not during model inference

---

## 🔊 Text-to-Speech (ElevenLabs)

- High-quality natural speech synthesis
- Converts refined sentences into audio
- API key injected securely via environment variables

---

## 🧰 Tech Stack

### 🔹 Backend & ML
- Python 3.9+
- Flask
- TensorFlow / Keras
- MediaPipe
- OpenCV
- NumPy
- Scikit-learn

### 🔹 NLP & Speech
- Groq API (LLM)
- ElevenLabs API (TTS)

### 🔹 DevOps
- Docker
- GitHub Actions
- AWS EC2
- AWS ECR
- IAM Roles

---

## 📁 Project Structure

sign-language-to-speech-conversion/
│
├── app.py # Flask inference server
├── extract_landmarks.py # MediaPipe landmark extraction
├── train_model.py # ANN training
├── modelnet_model.h5 # Trained model
├── scaler.pkl # Feature scaler
├── labels.json # Label mappings
│
├── data/ # Raw dataset
├── landmark_data/ # Extracted landmarks
│
├── templates/ # HTML templates
├── static/ # CSS / JS
│
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .gitignore
│
├── .github/workflows/
│ └── deploy.yml # CI/CD pipeline
│
└── README.md


---

## 🛠 Local Setup

### 1️⃣ Create Environment
```bash
conda create -n sign_lang python=3.9 -y
conda activate sign_lang
2️⃣ Install Dependencies
pip install -r requirements.txt
🎯 Landmark Extraction
python extract_landmarks.py
🎓 Train the ANN Model
python train_model.py
▶️ Run the Application
python app.py
Open:

http://127.0.0.1:5000
🐳 Docker Usage
Build Image
docker build -t sign-language-app .
Run Container
docker run -p 5000:5000 \
-e GROQ_API_KEY=your_key \
-e ELEVENLABS_API_KEY=your_key \
sign-language-app
☁️ AWS Deployment
Docker images stored in Amazon ECR

EC2 pulls images using IAM Role

App exposed via port 80

No SSH in CI/CD

🔁 CI/CD Pipeline
Trigger: git push to main

GitHub Actions:

Build Docker image

Push to ECR

EC2 auto-deploys latest image

📊 Performance (Observed)
Scenario	Accuracy
Images	95%+
Videos	90%+
Live Webcam	85–92%
🔐 Security Practices
❌ No API keys in code

✅ Environment variables only

✅ IAM Role for EC2

❌ No .pem in GitHub

✅ GitHub Secrets for CI/CD

👨‍💻 Author
Vivekananda Sahoo
Machine Learning Engineer
Deep Learning · Computer Vision · MLOps

⭐ Future Enhancements
Sentence-level temporal modeling (LSTM / Transformer)

Real-time streaming API

Mobile application

ONNX / TensorRT optimization




