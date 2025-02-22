# from flask import Flask, render_template, url_for

# app = Flask(__name__)  # Make sure this is present

# @app.route("/")
# def home():
#     return render_template('index.html')

# if __name__ == "__main__":
#     app.run(debug=True)

import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create folder if it doesn't exist

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
#app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB file size limit

# Allowed file extensions
# ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "pdf", "txt", "csv"}

# def allowed_file(filename):
#     """Check if file has an allowed extension"""
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"

        file = request.files["file"]

        if file.filename == "":
            return "No selected file"

        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            return redirect(url_for("index"))  # Refresh the page after upload

    uploaded_files = os.listdir(app.config["UPLOAD_FOLDER"])
    return render_template("index.html", files=uploaded_files)

if __name__ == "__main__":
    app.run(debug=True)

