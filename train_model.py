import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from collections import Counter
import joblib

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "landmark_data")

MODEL_PATH = os.path.join(BASE_DIR, "modelnet_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

EPOCHS = 80
BATCH_SIZE = 32

tf.random.set_seed(42)
np.random.seed(42)

# ---------------- LOAD DATA ----------------
X, y = [], []

LABELS = sorted(os.listdir(DATA_DIR))
label_to_index = {label: idx for idx, label in enumerate(LABELS)}

for label in LABELS:
    folder = os.path.join(DATA_DIR, label)
    for file in os.listdir(folder):
        data = np.load(os.path.join(folder, file))
        X.append(data)
        y.append(label_to_index[label])

X = np.array(X)
y = tf.keras.utils.to_categorical(y, num_classes=len(LABELS))

# 🔥 SHUFFLE DATA (VERY IMPORTANT)
X, y = shuffle(X, y, random_state=42)

print("✅ Loaded landmark dataset:", X.shape)

# 📊 CLASS BALANCE CHECK
print("📊 Samples per class:", Counter(np.argmax(y, axis=1)))

# ---------------- STANDARDIZE FEATURES ----------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

joblib.dump(scaler, SCALER_PATH)
print("✅ Scaler saved:", SCALER_PATH)

# ---------------- MODEL ----------------
model = Sequential([
    Dense(256, activation="relu", input_shape=(126,)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.25),

    Dense(64, activation="relu"),
    Dense(len(LABELS), activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------- CALLBACKS ----------------
early_stop = tf.keras.callbacks.EarlyStopping(
    patience=12,
    restore_best_weights=True
)

lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# ---------------- TRAIN ----------------
model.fit(
    X, y,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, lr_callback],
    verbose=1
)

# ---------------- SAVE ----------------
model.save(MODEL_PATH)
with open(LABELS_PATH, "w") as f:
    json.dump(LABELS, f)

print("✅ Model saved:", MODEL_PATH)
print("✅ Labels saved:", LABELS_PATH)
