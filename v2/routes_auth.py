from flask import Blueprint, request, jsonify
from .db import create_user, get_user

# ⭐ Blueprint (clean integration into app_v2)
auth_bp = Blueprint("auth_bp", __name__)


# ================= REGISTER =================
@auth_bp.route("/auth/register", methods=["POST"])
def register():
    data = request.get_json()

    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"success": False, "error": "Missing fields"}), 400

    user_id = create_user(username, password)

    if user_id is None:
        return jsonify({"success": False, "error": "User exists"}), 409

    return jsonify({
        "success": True,
        "user_id": user_id
    })


# ================= LOGIN =================
@auth_bp.route("/auth/login", methods=["POST"])
def login():
    data = request.get_json()

    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"success": False}), 400

    user = get_user(username, password)

    if not user:
        return jsonify({"success": False}), 401

    return jsonify({
        "success": True,
        "user_id": user["id"],
        "username": user["username"]
    })