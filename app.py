import os
import random
from flask import Flask, request, render_template, redirect, url_for, jsonify, send_from_directory, session
from werkzeug.utils import secure_filename
from processing.video_preprocessing import analyze_video
from processing.strAIn_Model import aggregate  # Ensure this function processes the CSV

app = Flask(__name__)
app.secret_key = "super_secret_key"  # Required for session handling

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def get_color(value):
    """Assign a color based on the strain intensity."""
    if value <= 0.2:
        return "#00FF00"  # Green (Low strain)
    if value <= 0.5:
        return "#FFFF00"  # Yellow (Medium strain)
    if value <= 0.8:
        return "#FFA500"  # Orange (High strain)
    return "#FF0000"  # Red (Very High strain)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "File not sent to server.", 400

        file = request.files["file"]
        weight = request.form.get("weight", "").strip()

        if not weight:
            return "Error: Weight is required!", 400
        if file.filename == "":
            return "No selected file", 400

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            # Remove any existing file in the uploads folder
            for existing_file in os.listdir(app.config["UPLOAD_FOLDER"]):
                existing_file_path = os.path.join(app.config["UPLOAD_FOLDER"], existing_file)
                os.remove(existing_file_path)

            file.save(file_path)

            # ✅ Process video and get CSV filename
            processed_video, csv_filename = analyze_video(file_path, app.config["UPLOAD_FOLDER"])
            
            # ✅ Compute aggregate strain data
            aggregate_data = aggregate(file_path, weight)

            # ✅ Store aggregate data in session
            session["aggregate_data"] = aggregate_data  # Flask handles it per user

            return redirect(url_for("view_video", filename=processed_video, csv_filename=csv_filename))

    return render_template("index.html")


@app.route("/view/<filename>")
def view_video(filename):
    """ Displays the processed video and provides a CSV download link """
    csv_filename = request.args.get("csv_filename")
    video_url = url_for("static", filename=f"uploads/{filename}")
    csv_url = url_for("download_csv", filename=csv_filename) if csv_filename else None

    return render_template("video.html", video_url=video_url, csv_url=csv_url)


@app.route("/download/<filename>")
def download_csv(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=True)


@app.route("/get_aggregate_data")
def get_aggregate_data():
    """ Returns the latest aggregate moment data for the heatmap """
    aggregate_data = session.get("aggregate_data", {})
    return jsonify(aggregate_data)


if __name__ == "__main__":
    app.run(debug=True)



# import os
# import random
# from flask import Flask, request, render_template, redirect, url_for, jsonify, send_from_directory
# from werkzeug.utils import secure_filename
# from processing.video_preprocessing import analyze_video
# from processing.strAIn_Model import aggregate

# app = Flask(__name__)

# UPLOAD_FOLDER = "static/uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# # Simulate model output (Replace with real ML model)

# # def get_model_output():
# #     return {
# #         "hip-left": round(random.uniform(0, 1), 2),
# #         "hip-right": round(random.uniform(0, 1), 2),
# #         "knee-left": round(random.uniform(0, 1), 2),
# #         "knee-right": round(random.uniform(0, 1), 2),
# #         "ankle-left": round(random.uniform(0, 1), 2),
# #         "ankle-right": round(random.uniform(0, 1), 2)
# #     }



# def get_model_output():
#     def get_color(value):
#         """Assign a color based on the strain intensity."""
#         if value <= 0.2:
#             return "#00FF00"  # Green (Low strain)
#         if value <= 0.5:
#             return "#FFFF00"  # Yellow (Medium strain)
#         if value <= 0.8:
#             return "#FFA500"  # Orange (High strain)
#         return "#FF0000"  # Red (Very High strain)

#     # Generate random strain values & assign colors
#     body_parts = ["hip-left", "hip-right", "knee-left", "knee-right", "ankle-left", "ankle-right"]
#     model_output = {}

#     for part in body_parts:
#         intensity = round(random.uniform(0, 1), 2)
#         model_output[part] = {"value": intensity, "color": get_color(intensity)}

#     return model_output


# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         if "file" not in request.files:
#             return "File not sent to server."

#         file = request.files["file"]

#         weight = request.form.get("weight", "").strip()  # Get weight input

#         if not weight:
#             return "Error: Weight is required!", 400
#         if file.filename == "":
#             return "No selected file"

#         if file:
#             filename = secure_filename(file.filename)
#             file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

#             # Remove any existing file in the uploads folder
#             for existing_file in os.listdir(app.config["UPLOAD_FOLDER"]):
#                 existing_file_path = os.path.join(app.config["UPLOAD_FOLDER"], existing_file)
#                 os.remove(existing_file_path)

#             file.save(file_path)

#             # ✅ Call analyze_video() and get BOTH filenames (video & CSV)
#             processed_video, csv_filename = analyze_video(file_path, app.config["UPLOAD_FOLDER"])
#             aggregate_data = analyze_video(file_path, app.config["UPLOAD_FOLDER"])

#             # ✅ Pass both filenames to view_video()
#             return redirect(url_for("view_video", filename=processed_video, csv_filename=csv_filename))

#     return render_template("index.html")

# @app.route("/view/<filename>")
# def view_video(filename):
#     """ Displays the processed video and provides a CSV download link """
#     csv_filename = request.args.get("csv_filename")  # Get CSV filename from URL args
#     video_url = url_for("static", filename=f"uploads/{filename}")
#     csv_url = url_for("download_csv", filename=csv_filename) if csv_filename else None

#     print(f"Serving video: {video_url}")  # Debugging log
#     return render_template("video.html", video_url=video_url, csv_url=csv_url)

# # ✅ Add a route to allow CSV download
# @app.route("/download/<filename>")
# def download_csv(filename):
#     return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=True)

# # API route to serve model output to JavaScript
# @app.route("/get_model_output")
# def get_model_data():
#     return jsonify(get_model_output())

# if __name__ == "__main__":
#     app.run(debug=True)
