import subprocess
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def analyze_video(video_path, output_folder):
    """
    Processes the uploaded video, extracts foot positions (X, Y), and saves:
    - A CSV file with motion data
    - A processed video with tracking overlay (H.264 format for browser compatibility)
    """
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {video_path}")
        return None, None

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if fps == 0:
        fps = 30  # Set default FPS if not detected

    # Filenames for processed video and CSV
    processed_video_filename = os.path.basename(video_path).replace(".mp4", "_processed.mp4")
    processed_video_path = os.path.join(output_folder, processed_video_filename)

    csv_filename = os.path.basename(video_path).replace(".mp4", "_foot_positions.csv")
    csv_path = os.path.join(output_folder, csv_filename)

    # OpenCV video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))

    # Create DataFrame for storing foot positions
    df = pd.DataFrame(columns=[
        "Frame", 
        "Left Heel X", "Left Heel Y", "Left Foot Index X", "Left Foot Index Y",
        "Right Heel X", "Right Heel Y", "Right Foot Index X", "Right Foot Index Y"
    ])

    frame_num = 0
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"⚠️ Warning: Could not read frame {frame_num}, stopping processing.")
                break

            # Convert image to RGB for Mediapipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract foot positions if detected
            try:
                landmarks = results.pose_landmarks.landmark

                left_heel_x = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x
                left_heel_y = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y
                left_foot_index_x = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x
                left_foot_index_y = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y

                right_heel_x = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x
                right_heel_y = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y
                right_foot_index_x = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x
                right_foot_index_y = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y

                # Append to DataFrame
                df.loc[frame_num] = [
                    frame_num,
                    left_heel_x, left_heel_y, left_foot_index_x, left_foot_index_y,
                    right_heel_x, right_heel_y, right_foot_index_x, right_foot_index_y
                ]

                # Draw pose landmarks
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

            except Exception:
                print(f"⚠️ Frame {frame_num}: No foot landmarks detected.")

            # Write processed frame to video
            out.write(image)
            frame_num += 1

    cap.release()
    out.release()

    # Save DataFrame to CSV
    df.to_csv(csv_path, index=False)
    print(f"✅ Foot position data saved to: {csv_path}")

    # ✅ Step 2: Convert the processed video to H.264 using FFmpeg
    fixed_video_filename = processed_video_filename.replace(".mp4", "_fixed.mp4")
    fixed_video_path = os.path.join(output_folder, fixed_video_filename)

    ffmpeg_command = [
        "ffmpeg", "-y", "-i", processed_video_path,  # Input video
        "-c:v", "libx264", "-preset", "slow", "-crf", "22",  # H.264 encoding
        "-c:a", "aac", "-b:a", "128k",  # Audio settings
        fixed_video_path  # Output video
    ]

    try:
        subprocess.run(ffmpeg_command, check=True)
        print(f"✅ Re-encoded video saved: {fixed_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during FFmpeg conversion: {e}")
        return None, None

    # ✅ Step 3: Return the fixed video and CSV filenames
    return fixed_video_filename, csv_filename
