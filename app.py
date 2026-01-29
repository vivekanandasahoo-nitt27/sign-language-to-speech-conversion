import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# ===== IMPORT YOUR WORKING FUNCTIONS =====
from predictor import predict_video, predict_image
# from speech import SpeechEngine
# speaker = SpeechEngine(rate=170, volume=1.0, cooldown=2.0)


# ===== CONFIG =====
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ===== HELPERS =====
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ===== ROUTES =====
@app.route("/")
def index():
    return render_template("index.html")


# ---------- IMAGE (upload + webcam frames) ----------
@app.route("/predict_image", methods=["POST"])
def predict_image_route():
    if "file" not in request.files:
        return jsonify({"error": "No file received"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
    file.save(path)

    prediction = predict_image(path)
    os.remove(path)

    #speaker.speak(prediction)
    # if prediction not in ["Uncertain", "No hand detected"]:
    #     try:
    #         speaker.speak(prediction)
    #     except Exception as e:
    #         print("🔇 Speech error:", e)
    return jsonify({"prediction": prediction})



# ---------- VIDEO ----------
@app.route("/predict_video", methods=["POST"])
def predict_video_route():
    print("🎥 Video route hit")   # ADD THIS

    if "file" not in request.files:
        print("❌ No file in request.files")
        return jsonify({"error": "No file received"}), 400

    file = request.files["file"]
    print("📁 Received file:", file.filename)  # ADD THIS

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
    file.save(path)

    print("✅ Saved video to:", path)  # ADD THIS

    prediction = predict_video(path)
    os.remove(path)
    # if prediction not in ["No signs detected"]:
    #     try:
    #         speaker.speak(prediction)
    #     except Exception as e:
    #         print("🔇 Speech error:", e)


    return jsonify({"prediction": prediction})

# ===== RUN =====
if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False
    )
