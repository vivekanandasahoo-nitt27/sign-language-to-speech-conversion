🤟 Sign Language to Speech Conversion System

MediaPipe + ANN + Groq NLP + ElevenLabs + Flask + Docker + AWS CI/CD

A full-stack Sign Language to Speech web application that detects hand gestures from images, videos, or live webcam input, converts them into text using hand landmark–based ANN classification, refines sentences using Groq LLM, and finally converts text into natural speech using ElevenLabs.

Designed with scalability, accuracy, and production deployment in mind.

🚀 Key Highlights

✋ MediaPipe Hand Landmark Extraction (2 hands supported)

🧠 ANN-based gesture classification (landmark-driven, not raw images)

📐 Normalized & scale-invariant hand pose features

🧠 Groq LLM for word & sentence refinement

🔊 ElevenLabs text-to-speech output

🌐 Flask-based web application

🐳 Dockerized deployment

☁️ AWS EC2 + ECR + GitHub Actions CI/CD

🔐 Secure API key handling via environment variables

🧠 System Architecture (Updated)
Input (Image / Video / Webcam)
        ↓
MediaPipe Hands
(21 landmarks × 2 hands)
        ↓
Landmark Normalization
- Wrist-relative
- Unit-scale normalization
- Zero-padding for single hand
        ↓
ANN Classifier (126 features)
        ↓
Raw Text Prediction
        ↓
Groq LLM (NLP Refinement)
        ↓
Grammatically Correct Sentence
        ↓
ElevenLabs TTS
        ↓
Speech Output 🔊

🧩 Landmark Extraction Pipeline

Detect up to 2 hands per frame

Extract 21 landmarks per hand (x, y, z)

Normalize landmarks:

Relative to wrist

Scale to unit distance

Pad missing hand with zeros

Final feature vector:

2 hands × 21 landmarks × 3 = 126 features


📁 Output stored as .npy files per label.

🧠 ANN Model Architecture
Input: 126 landmark features
↓
Dense(256) + BatchNorm + Dropout(0.3)
↓
Dense(128) + BatchNorm + Dropout(0.25)
↓
Dense(64)
↓
Dense(N classes) + Softmax

Training Configuration
Parameter	Value
Epochs	80
Batch Size	32
Optimizer	Adam
Learning Rate	1e-3
Loss	Categorical Crossentropy
Validation Split	20%
Callbacks	EarlyStopping, ReduceLROnPlateau
🧠 NLP with Groq

Refines:

Broken words

Incomplete sequences

Contextual meaning

Converts gesture outputs into human-readable sentences

Integrated after prediction, not during classification

🔊 Text-to-Speech (ElevenLabs)

Converts refined text to natural speech

High-quality voice synthesis

API key injected securely via environment variables

🧰 Tech Stack
Backend & ML

Python 3.9+

Flask

TensorFlow / Keras

MediaPipe

OpenCV

NumPy

Scikit-learn

NLP & Speech

Groq API (LLM)

ElevenLabs API (TTS)

DevOps

Docker

GitHub Actions (CI/CD)

AWS EC2

AWS ECR

IAM Roles (no hardcoded AWS keys)

📁 Project Structure
sign-language-to-speech-conversion/
│
├── app.py                    # Flask inference server
├── extract_landmarks.py      # MediaPipe landmark extraction
├── train_model.py            # ANN training script
├── modelnet_model.h5         # Trained ANN model
├── scaler.pkl                # Feature standard scaler
├── labels.json               # Class labels
│
├── landmark_data/            # Extracted landmark features
├── data/                     # Raw image dataset
│
├── templates/                # HTML templates
├── static/                   # CSS / JS assets
│
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .gitignore
│
├── .github/workflows/
│   └── deploy.yml             # GitHub Actions CI/CD
│
└── README.md

🛠 Local Setup
1️⃣ Create Environment
conda create -n sign_lang python=3.9 -y
conda activate sign_lang

2️⃣ Install Dependencies
pip install -r requirements.txt

🎯 Landmark Extraction
python extract_landmarks.py


Outputs:

landmark_data/

labels.json

🎓 Train the ANN Model
python train_model.py


Outputs:

modelnet_model.h5

scaler.pkl

labels.json

▶️ Run the Application
python app.py


Open in browser:

http://127.0.0.1:5000

🐳 Docker Usage
Build Image
docker build -t sign-language-app .

Run Container
docker run -p 5000:5000 \
-e GROQ_API_KEY=your_key \
-e ELEVENLABS_API_KEY=your_key \
sign-language-app

☁️ AWS Deployment (EC2 + ECR)

Docker image pushed to Amazon ECR

EC2 instance pulls image using IAM Role

App runs on port 80

CI/CD handled via GitHub Actions

🔁 CI/CD Pipeline (GitHub Actions)

Trigger: git push to main

Steps:

Checkout code

Build Docker image

Push to Amazon ECR

EC2 auto-deploys latest image

No SSH. No .pem in GitHub. Secure & scalable.

📊 Performance (Observed)
Metric	Accuracy
Image-based prediction	95%+
Video prediction	90%+
Real-world webcam	85–92%
🔐 Security Best Practices

❌ No API keys in code

✅ Environment variables only

✅ IAM Roles for EC2

❌ No .pem keys in GitHub

✅ Secrets managed via GitHub Actions

👨‍💻 Author

Vivekananda Sahoo
Machine Learning Engineer
Deep Learning • Computer Vision • MLOps

⭐ Future Enhancements

Sentence-level temporal modeling (LSTM / Transformer)

Real-time streaming API

Mobile app (Flutter / React Native)

ONNX / TensorRT optimization

GPU-based EC2 inference

If you want, next I can:

✨ Optimize this for resume / LinkedIn

📉 Reduce Docker image size

🔁 Add versioned rollback

📊 Add monitoring & logs

Just tell me 👌