import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)


UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

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

if __name__ == "__main__":
    app.run(debug=True)

