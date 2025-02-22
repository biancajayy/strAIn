import os
import random
from flask import Flask, request, render_template, redirect, url_for, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)


UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Simulate model output (Replace with real ML model)
def get_model_output():
    # see json at http://127.0.0.1:5000/get_model_output
    return {
        "shoulder-left": round(random.uniform(0, 1), 2),
        "shoulder-right": round(random.uniform(0, 1), 2),
        "elbow-left": round(random.uniform(0, 1), 2),
        "elbow-right": round(random.uniform(0, 1), 2),
        "hip-left": round(random.uniform(0, 1), 2),
        "hip-right": round(random.uniform(0, 1), 2),
        "knee-left": round(random.uniform(0, 1), 2),
        "knee-right": round(random.uniform(0, 1), 2),
        "ankle-left": round(random.uniform(0, 1), 2),
        "ankle-right": round(random.uniform(0, 1), 2)
    }

#TODO: possibly set up file restrictions/ how many uploads at once

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "File not sent to server."

        file = request.files["file"]

        if file.filename == "":
            return "No selected file"

        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            return redirect(url_for("view_video", filename=filename)) # redirect to video page w results
    return render_template("index.html")
        
@app.route("/view/<filename>")
def view_video(filename):
    video_url = url_for("static", filename=f"uploads/{filename}")
    return render_template("video.html", video_url=video_url)

# API route to serve model output to JavaScript
@app.route("/get_model_output")
def get_model_data():
    return jsonify(get_model_output())

if __name__ == "__main__":
    app.run(debug=True)

