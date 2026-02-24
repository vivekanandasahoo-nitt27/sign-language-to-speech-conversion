from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename

# ⭐ original speech (unchanged)
from .speech_service import text_to_speech_v2


from flask import Flask, request, jsonify, render_template, session, redirect, url_for

app = Flask(__name__, template_folder="templates", static_folder="../static")
app.secret_key = "dev-secret-key-123"   # required for session

# ⭐ v2 modules
from .db import init_db
from .routes_auth import auth_bp
from .predictor_adapter import PredictorAdapter

# ================= INIT =================
init_db()


app.register_blueprint(auth_bp)

adapter = PredictorAdapter()

UPLOAD_FOLDER = "uploads_v2"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



@app.route("/")
def main_ui():
    if not session.get("user_id"):
        return redirect(url_for("login_page"))
    return render_template("index.html")



@app.route("/logout")
def logout():
    session.clear()                     # remove login
    return redirect(url_for("login_page"))   # go to login page


# ================= LOGIN PAGE =================
@app.route("/login")
def login_page():
    return render_template("login.html")



# after login 
@app.route("/after_login")
def after_login():
    user_id = request.args.get("user_id")

    if not user_id:
        return redirect(url_for("login_page"))

    session["user_id"] = int(user_id)
    return redirect(url_for("main_ui"))


# ================= VIDEO V2 =================
@app.route("/predict_video_v2", methods=["POST"])
def predict_video_v2():
    user_id = session.get("user_id")
    if not user_id:
        return {"error": "Not logged in"}, 401

    file = request.files.get("file")
    if not file:
        return {"error": "No file"}, 400

    path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(path)

    # ⭐ adapter does segmentation + NLP + memory
    sentences = adapter.process_video(path, int(user_id))

    os.remove(path)

    # ⭐ speech
    audio_list = []
    for s in sentences:
        audio = text_to_speech_v2(s)
        audio_list.append(audio)

    return jsonify({
        "sentences": sentences,
        "audio": audio_list
    })


# ================= RUN =================
if __name__ == "__main__":
    print("🚀 Starting V2 app...")
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)