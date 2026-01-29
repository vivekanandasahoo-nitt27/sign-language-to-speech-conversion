# 🤟 Sign Language Detection Web App  
**CNN + Flask + MediaPipe + Docker + AWS CI/CD**

A full-stack deep learning web application that translates hand sign images, videos, or live webcam input into text using a Convolutional Neural Network (CNN).  
Built with TensorFlow, MediaPipe, Flask, Docker, and designed for AWS deployment with CI/CD.

---

## 🚀 Features

- 📂 Upload hand sign **images or videos**
- 🎥 **Live webcam** sign detection
- 🧠 CNN-based sign classification (A–Z, 0–9)
- ✋ MediaPipe hand detection
- 🌐 REST API using Flask
- 🐳 Dockerized for easy deployment
- ☁️ AWS + GitHub Actions CI/CD ready

---

## 🧠 CNN Model Architecture

Input (128×128×3)
↓
Conv2D(32) + BatchNorm + MaxPooling
↓
Conv2D(64) + BatchNorm + MaxPooling
↓
Conv2D(128) + BatchNorm + MaxPooling
↓
Conv2D(256) + BatchNorm + MaxPooling
↓
GlobalAveragePooling
↓
Dense(256) + Dropout(0.5)
↓
Dense(36) → Softmax Output


### Training Settings
| Parameter         | Value           |
|------------------|-----------------|
| Image Size       | 128 × 128       |
| Optimizer        | Adam            |
| Learning Rate    | 1e-4            |
| Loss             | Categorical CE  |
| Epochs           | 70              |
| Batch Size       | 32              |
| Augmentation     | Rotation, Zoom, Shift, Brightness, Flip |
| Callbacks        | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |

---

## 🧰 Tech Stack

### Backend
- Python 3.9  
- Flask  
- TensorFlow / Keras  
- MediaPipe  
- OpenCV  
- NumPy  

### Frontend
- HTML5  
- CSS3  
- JavaScript  

### ML
- CNN (Convolutional Neural Network)
- ImageDataGenerator
- MediaPipe Hands

### DevOps
- Docker  
- GitHub Actions  
- AWS EC2 / ECS / ECR  
- Nginx (optional)

---

## 📁 Project Structure

sign-language-detector/
│
├── app.py # Flask inference server
├── train_app.py # CNN training script
├── modelnet_model.h5 # Trained model
├── labels.json # Label mappings
│
├── data/ # Training dataset
│ ├── a/
│ ├── b/
│ └── ...
│
├── templates/
│ └── index.html # Frontend UI
│
├── static/
│ ├── style.css # Styling
│ └── script.js # Frontend logic
│
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md


---

## 🛠 Local Setup

### 1️⃣ Create Environment
```bash
conda create -n hand_sign python=3.9 -y
conda activate hand_sign
2️⃣ Install Dependencies
pip install -r requirements.txt
🎯 Training the CNN
python train_app.py
Outputs:

modelnet_model.h5

labels.json

▶️ Run the Web App
python app.py
Open in browser:

http://127.0.0.1:5000
🐳 Docker Setup
Build Image
docker build -t hand-sign-app .
Run Container
docker run -p 5000:5000 hand-sign-app
☁️ AWS Deployment (EC2 + Docker)
Push image to ECR

Launch EC2 instance

Install Docker

Pull image from ECR

Run container

docker run -d -p 80:5000 hand-sign-app
🔁 CI/CD (GitHub Actions)
.github/workflows/deploy.yml

name: Deploy to AWS

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Login to ECR
        uses: aws-actions/amazon-ecr-login@v1

      - name: Build & Push Image
        run: |
          docker build -t hand-sign-app .
          docker tag hand-sign-app:latest <ECR_URL>:latest
          docker push <ECR_URL>:latest
📊 Expected Performance
Metric	Value
Training Accuracy	95–98%
Validation Accuracy	92–96%
Real-world Accuracy	85–92%
⚠️ Notes
Ensure IMG_SIZE in app.py matches training (128).

Disable MediaPipe crop when using pre-cropped images.

Use GPU TensorFlow for faster training.

For production, replace Flask dev server with Gunicorn.

👨‍💻 Author
Vivekananda Sahoo
ML Engineer | Deep Learning | Computer Vision

⭐ Future Enhancements
LSTM for sentence prediction

Transformer-based sign NLP

Mobile app (Flutter)

ONNX model export

Realtime streaming API

