# app.py
from flask import Flask, jsonify, request, send_from_directory
import os
from model import MiningDetectionModel
from utils import load_image, preprocess

app = Flask(__name__, static_folder="static")

model = MiningDetectionModel()

@app.route("/")
def home():
    return send_from_directory("static", "minewatch_v3.html")

@app.route("/detect", methods=["POST"])
def detect():
    before_path = "data/before.png"
    after_path = "data/after.png"

    before = preprocess(load_image(before_path))
    after = preprocess(load_image(after_path))

    score = model.predict(before, after)

    result = {
        "zone": "Zone A-7",
        "confidence": round(score, 2),
        "risk": "HIGH" if score > 0.7 else "MEDIUM" if score > 0.4 else "LOW",
        "area": "2.4 hectares"
    }

    return jsonify(result)

@app.route("/alerts")
def alerts():
    # Dummy alerts (can be replaced later)
    return jsonify([
        {"id": "A1", "score": 0.91, "risk": "HIGH"},
        {"id": "A2", "score": 0.65, "risk": "MEDIUM"},
        {"id": "A3", "score": 0.22, "risk": "LOW"}
    ])

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
