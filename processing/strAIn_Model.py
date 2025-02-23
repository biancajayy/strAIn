import cv2
import os
import numpy as np
import mediapipe as mp
import open3d as o3d
import time
import pandas as pd
import joblib
import xgboost as xgb


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def aggregate(vid_to_analyze, body_weight):
    columns = pd.MultiIndex.from_tuples([
        ('Left Hip', 'x'), ('Left Hip', 'y'), ('Left Hip', 'Angle'),
        ('Right Hip', 'x'), ('Right Hip', 'y'), ('Right Hip', 'Angle'),
        ('Left Knee', 'x'), ('Left Knee', 'y'), ('Left Knee', 'Angle'),
        ('Right Knee', 'x'), ('Right Knee', 'y'), ('Right Knee', 'Angle'),
        ('Left Ankle', 'x'), ('Left Ankle', 'y'), ('Left Ankle', 'Angle'),
        ('Right Ankle', 'x'), ('Right Ankle', 'y'), ('Right Ankle', 'Angle'),
        ('Left Foot Middle', 'x'), ('Left Foot Middle', 'y'),
        ('Right Foot Middle', 'x'), ('Right Foot Middle', 'y')
    ])
    motion_df = pd.DataFrame(columns=columns)
    motion_df.index.name = 'Frame'

    frame_num = 0

    cap = cv2.VideoCapture(vid_to_analyze)
    
    # Read the first frame to determine image dimensions
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Unable to read video frames; check your video file.")
    image_height, image_width = frame.shape[:2]
    
    # Reset the video capture to the first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Recolor Image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make Detection
            results = pose.process(image)

            # Recolor image back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract Landmarks and Process
            try:
                landmarks = results.pose_landmarks.landmark
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
                left_foot = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y,
                             landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z]
                right_foot = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y,
                              landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z]
                left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y,
                             landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].z]
                right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y,
                              landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].z]

                # Calculate Angle
                left_shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
                right_shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)
                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                left_hip_angle = calculate_angle(left_knee, left_hip, left_shoulder)
                right_hip_angle = calculate_angle(right_knee, right_hip, right_shoulder)
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                left_ankle_angle = calculate_angle(left_knee, left_ankle, left_foot)
                right_ankle_angle = calculate_angle(right_knee, right_ankle, right_foot)
                left_shoulder_secondary_angle = calculate_angle(left_elbow, left_shoulder, right_shoulder)
                right_shoulder_secondary_angle = calculate_angle(right_elbow, right_shoulder, left_shoulder)
                left_hip_secondary_angle = calculate_angle(left_knee, left_hip, right_hip)
                right_hip_secondary_angle = calculate_angle(right_knee, right_hip, left_hip)

                # Calculate Foot Mid Point
                left_foot_middle = [
                    (left_heel[0] + left_foot[0]) / 2,
                    (left_heel[1] + left_foot[1]) / 2,
                    (left_heel[2] + left_foot[2]) / 2
                ]
                right_foot_middle = [
                    (right_heel[0] + right_foot[0]) / 2,
                    (right_heel[1] + right_foot[1]) / 2,
                    (right_heel[2] + right_foot[2]) / 2
                ]

                # Construct Motion Data
                row_data = {
                    ('Left Hip', 'x'): left_hip[0],
                    ('Left Hip', 'y'): left_hip[1],
                    ('Left Hip', 'Angle'): left_hip_angle,
                    ('Left Hip', 'Secondary Angle'): left_hip_secondary_angle,
                    ('Right Hip', 'x'): right_hip[0],
                    ('Right Hip', 'y'): right_hip[1],
                    ('Right Hip', 'Angle'): right_hip_angle,
                    ('Right Hip', 'Secondary Angle'): right_hip_secondary_angle,
                    ('Left Knee', 'x'): left_knee[0],
                    ('Left Knee', 'y'): left_knee[1],
                    ('Left Knee', 'Angle'): left_knee_angle,
                    ('Right Knee', 'x'): right_knee[0],
                    ('Right Knee', 'y'): right_knee[1],
                    ('Right Knee', 'Angle'): right_knee_angle,
                    ('Left Ankle', 'x'): left_ankle[0],
                    ('Left Ankle', 'y'): left_ankle[1],
                    ('Left Ankle', 'Angle'): left_ankle_angle,
                    ('Right Ankle', 'x'): right_ankle[0],
                    ('Right Ankle', 'y'): right_ankle[1],
                    ('Right Ankle', 'Angle'): right_ankle_angle,
                    ('Left Foot Middle', 'x'): left_foot_middle[0],
                    ('Left Foot Middle', 'y'): left_foot_middle[1],
                    ('Right Foot Middle', 'x'): right_foot_middle[0],
                    ('Right Foot Middle', 'y'): right_foot_middle[1]
                }

                motion_df.loc[frame_num] = row_data
                frame_num += 1

            except Exception as e:
                # You can log the exception here if needed
                pass

            # Render Detections
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

            #cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    # Use image dimensions to un-normalize landmark data
    motion_df[('Left Hip', 'x')] = motion_df[('Left Hip', 'x')] * image_width
    motion_df[('Left Hip', 'y')] = motion_df[('Left Hip', 'y')] * image_height
    motion_df[('Right Hip', 'x')] = motion_df[('Right Hip', 'x')] * image_width
    motion_df[('Right Hip', 'y')] = motion_df[('Right Hip', 'y')] * image_height
    motion_df[('Left Knee', 'x')] = motion_df[('Left Knee', 'x')] * image_width
    motion_df[('Left Knee', 'y')] = motion_df[('Left Knee', 'y')] * image_height
    motion_df[('Right Knee', 'x')] = motion_df[('Right Knee', 'x')] * image_width
    motion_df[('Right Knee', 'y')] = motion_df[('Right Knee', 'y')] * image_height
    motion_df[('Left Ankle', 'x')] = motion_df[('Left Ankle', 'x')] * image_width
    motion_df[('Left Ankle', 'y')] = motion_df[('Left Ankle', 'y')] * image_height
    motion_df[('Right Ankle', 'x')] = motion_df[('Right Ankle', 'x')] * image_width
    motion_df[('Right Ankle', 'y')] = motion_df[('Right Ankle', 'y')] * image_height
    motion_df[('Left Foot Middle', 'x')] = motion_df[('Left Foot Middle', 'x')] * image_width
    motion_df[('Left Foot Middle', 'y')] = motion_df[('Left Foot Middle', 'y')] * image_height
    motion_df[('Right Foot Middle', 'x')] = motion_df[('Right Foot Middle', 'x')] * image_width
    motion_df[('Right Foot Middle', 'y')] = motion_df[('Right Foot Middle', 'y')] * image_height

    # Set Velocity and Acceleration Columns for Each Landmark
    for landmark in ['Left Hip', 'Right Hip', 'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle']:
        angular_velocity = motion_df[(landmark, 'Angle')].diff()
        angular_acceleration = np.deg2rad(angular_velocity.diff())
        motion_df[(landmark, 'Angular Acceleration')] = angular_acceleration
    for landmark in ['Left Foot Middle', 'Right Foot Middle']:
        y_col = motion_df[(landmark, 'y')]
        x_col = motion_df[(landmark, 'x')]
        vel_x = x_col.diff()
        vel_y = y_col.diff()
        acc_x = vel_x.diff()
        acc_y = vel_y.diff()
        motion_df[(landmark, 'Vel_x')] = vel_x / 60
        motion_df[(landmark, 'Vel_y')] = vel_y / 60 
        motion_df[(landmark, 'Acc_x')] = acc_x / 60
        motion_df[(landmark, 'Acc_y')] = acc_y / 60

    # Interpolate Missing Values
    cleaned_motion_df = motion_df.interpolate(method='linear', limit_direction='forward')
    cleaned_motion_df.fillna(0, inplace=True)

    current_directory = os.getcwd()
    print("Current working directory:", current_directory)
    model = joblib.load('processing/xgb_prediction_model.pkl')

    features = np.column_stack([
        cleaned_motion_df[('Left Foot Middle', 'x')].values,
        cleaned_motion_df[('Left Foot Middle', 'y')].values,
        cleaned_motion_df[('Right Foot Middle', 'x')].values,
        cleaned_motion_df[('Right Foot Middle', 'y')].values,
        cleaned_motion_df[('Left Foot Middle', 'Vel_x')].values,
        cleaned_motion_df[('Left Foot Middle', 'Vel_y')].values,
        cleaned_motion_df[('Right Foot Middle', 'Vel_x')].values,
        cleaned_motion_df[('Right Foot Middle', 'Vel_y')].values,
        cleaned_motion_df[('Left Foot Middle', 'Acc_x')].values,
        cleaned_motion_df[('Left Foot Middle', 'Acc_y')].values,
        cleaned_motion_df[('Right Foot Middle', 'Acc_x')].values,
        cleaned_motion_df[('Right Foot Middle', 'Acc_y')].values
    ])

    predictions = model.predict(features)

    cleaned_motion_df['GRF_L_x'] = predictions[:, 0]
    cleaned_motion_df['GRF_L_y'] = predictions[:, 1]
    cleaned_motion_df['GRF_R_x'] = predictions[:, 2]
    cleaned_motion_df['GRF_R_y'] = predictions[:, 3]

    # Calculate Joint Moments
    body_mass = float(body_weight)
    body_mass /= 2.20462
    shank_mass = 0.0457 * body_mass
    thigh_mass = 0.1447 * body_mass
    foot_mass = 0.0133 * body_mass
    g = 9.81

    shank_length = np.linalg.norm(
        np.array([cleaned_motion_df[('Left Knee', 'x')], cleaned_motion_df[('Left Knee', 'y')]]) -
        np.array([cleaned_motion_df[('Left Ankle', 'x')], cleaned_motion_df[('Left Ankle', 'y')]])
    )
    thigh_length = np.linalg.norm(
        np.array([cleaned_motion_df[('Left Hip', 'x')], cleaned_motion_df[('Left Hip', 'y')]]) -
        np.array([cleaned_motion_df[('Left Knee', 'x')], cleaned_motion_df[('Left Knee', 'y')]])
    )
    foot_length = np.linalg.norm(
        np.array([cleaned_motion_df[('Left Ankle', 'x')], cleaned_motion_df[('Left Ankle', 'y')]]) -
        np.array([cleaned_motion_df[('Left Foot Middle', 'x')], cleaned_motion_df[('Left Foot Middle', 'y')]])
    )

    I_foot = 1 / 3 * foot_mass * foot_length ** 2
    I_shank = 1 / 3 * shank_mass * shank_length ** 2
    I_thigh = 1 / 3 * thigh_mass * thigh_length ** 2

    net_moments = []

    for index, row in cleaned_motion_df.iterrows():
        left_hip = np.array([row[('Left Hip', 'x')], row[('Left Hip', 'y')]])
        left_knee = np.array([row[('Left Knee', 'x')], row[('Left Knee', 'y')]])
        left_ankle = np.array([row[('Left Ankle', 'x')], row[('Left Ankle', 'y')]])
        left_foot = np.array([row[('Left Foot Middle', 'x')], row[('Left Foot Middle', 'y')]])

        right_hip = np.array([row[('Right Hip', 'x')], row[('Right Hip', 'y')]])
        right_knee = np.array([row[('Right Knee', 'x')], row[('Right Knee', 'y')]])
        right_ankle = np.array([row[('Right Ankle', 'x')], row[('Right Ankle', 'y')]])
        right_foot = np.array([row[('Right Foot Middle', 'x')], row[('Right Foot Middle', 'y')]])

        left_shank_CoM = (left_knee + left_ankle) / 2
        left_thigh_CoM = (left_hip + left_knee) / 2

        right_shank_CoM = (right_knee + right_ankle) / 2
        right_thigh_CoM = (right_hip + right_knee) / 2

        F_foot = np.array([row['GRF_L_x'], row['GRF_L_y']])
        alpha_ankle = row[('Left Ankle', 'Angular Acceleration')]
        moment_ankle_inertial = I_foot * alpha_ankle
        r_ankle = left_foot - left_ankle
        moment_ankle = r_ankle[0] * F_foot[1] - r_ankle[1] * F_foot[0]
        r_knee = left_ankle - left_knee
        moment_knee_external = r_knee[0] * F_foot[1] - r_knee[1] * F_foot[0]
        alpha_shank = row[('Left Knee', 'Angular Acceleration')]
        moment_knee_inertial = I_shank * alpha_shank
        r_hip = left_knee - left_hip
        moment_hip_external = r_hip[0] * F_foot[1] - r_hip[1] * F_foot[0]
        alpha_thigh = row[('Left Hip', 'Angular Acceleration')]
        moment_hip_inertial = I_thigh * alpha_thigh

        r_gravity_foot = left_ankle - left_foot
        r_gravity_shank = left_knee - left_shank_CoM
        r_gravity_thigh = left_hip - left_thigh_CoM

        F_gravity_foot = np.array([0, -foot_mass * g])
        F_gravity_shank = np.array([0, -shank_mass * g])
        F_gravity_thigh = np.array([0, -thigh_mass * g])

        moment_gravity_foot = r_gravity_foot[0] * F_gravity_foot[1] - r_gravity_foot[1] * F_gravity_foot[0]
        moment_gravity_shank = r_gravity_shank[0] * F_gravity_shank[1] - r_gravity_shank[1] * F_gravity_shank[0]
        moment_gravity_thigh = r_gravity_thigh[0] * F_gravity_thigh[1] - r_gravity_thigh[1] * F_gravity_thigh[0]

        net_moment_left_ankle = moment_ankle + moment_ankle_inertial + moment_gravity_foot
        net_moment_left_knee = moment_knee_external + moment_knee_inertial + moment_gravity_shank
        net_moment_left_hip = moment_hip_external + moment_hip_inertial + moment_gravity_thigh

        F_foot = np.array([row['GRF_R_x'], row['GRF_R_y']])
        alpha_ankle = row[('Right Ankle', 'Angular Acceleration')]
        moment_ankle_inertial = I_foot * alpha_ankle
        r_ankle = right_foot - right_ankle
        moment_ankle = r_ankle[0] * F_foot[1] - r_ankle[1] * F_foot[0]
        r_knee = right_ankle - right_knee
        moment_knee_external = r_knee[0] * F_foot[1] - r_knee[1] * F_foot[0]
        alpha_shank = row[('Right Knee', 'Angular Acceleration')]
        moment_knee_inertial = I_shank * alpha_shank
        r_hip = right_knee - right_hip
        moment_hip_external = r_hip[0] * F_foot[1] - r_hip[1] * F_foot[0]
        alpha_thigh = row[('Right Hip', 'Angular Acceleration')]
        moment_hip_inertial = I_thigh * alpha_thigh

        r_gravity_foot = right_ankle - right_foot
        r_gravity_shank = right_knee - right_shank_CoM
        r_gravity_thigh = right_hip - right_thigh_CoM

        F_gravity_foot = np.array([0, -foot_mass * g])
        F_gravity_shank = np.array([0, -shank_mass * g])
        F_gravity_thigh = np.array([0, -thigh_mass * g])

        moment_gravity_foot = r_gravity_foot[0] * F_gravity_foot[1] - r_gravity_foot[1] * F_gravity_foot[0]
        moment_gravity_shank = r_gravity_shank[0] * F_gravity_shank[1] - r_gravity_shank[1] * F_gravity_shank[0]
        moment_gravity_thigh = r_gravity_thigh[0] * F_gravity_thigh[1] - r_gravity_thigh[1] * F_gravity_thigh[0]

        net_moment_right_ankle = moment_ankle + moment_ankle_inertial + moment_gravity_foot
        net_moment_right_knee = moment_knee_external + moment_knee_inertial + moment_gravity_shank
        net_moment_right_hip = moment_hip_external + moment_hip_inertial + moment_gravity_thigh

        net_moments.append({
            'Frame': index,
            'Left_Ankle_Moment': net_moment_left_ankle,
            'Left_Knee_Moment': net_moment_left_knee,
            'Left_Hip_Moment': net_moment_left_hip,
            'Right_Ankle_Moment': net_moment_right_ankle,
            'Right_Knee_Moment': net_moment_right_knee,
            'Right_Hip_Moment': net_moment_right_hip
        })
        


    moments_df = pd.DataFrame(net_moments)
    total_frames = len(moments_df)
    aggregate_moment = pd.DataFrame(moments_df.sum().abs()).transpose().drop('Frame', axis=1).astype('int').apply(lambda x : x / (total_frames * 60)).to_dict()
    

    print(aggregate_moment)

    return aggregate_moment

    # moments_df = pd.DataFrame(net_moments)
    # aggregate_moment = pd.DataFrame(moments_df.abs().sum()).transpose().drop('Frame', axis=1)

    # aggregate_moment = aggregate_moment.to_dict()
    
    # return aggregate_moment

    # # Create the moments DataFrame

