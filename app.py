import cv2
import mediapipe as mp
import numpy as np
import requests
import tempfile
import os
from flask import Flask, request, jsonify

app = Flask(__name__)
mp_pose = mp.solutions.pose

def angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

def lm(landmarks, idx):
    p = landmarks[idx]
    return [p.x, p.y, p.z]

def spine_angle(landmarks):
    L = landmarks
    shoulder_mid = np.mean([lm(L,11), lm(L,12)], axis=0)
    hip_mid      = np.mean([lm(L,23), lm(L,24)], axis=0)
    vertical_ref = hip_mid + np.array([0, -0.2, 0])
    return float(angle(shoulder_mid, hip_mid, vertical_ref))

def shoulder_abduction(landmarks, side="left"):
    L = landmarks
    if side == "left":
        return angle(lm(L,13), lm(L,11), lm(L,23))
    else:
        return angle(lm(L,14), lm(L,12), lm(L,24))

def trunk_rotation(landmarks):
    L = landmarks
    ls, rs = np.array(lm(L,11)), np.array(lm(L,12))
    lh, rh = np.array(lm(L,23)), np.array(lm(L,24))
    shoulder_vec = rs - ls
    hip_vec      = rh - lh
    s2d = np.array([shoulder_vec[0], shoulder_vec[2]])
    h2d = np.array([hip_vec[0],      hip_vec[2]])
    cos = np.dot(s2d, h2d) / (np.linalg.norm(s2d) * np.linalg.norm(h2d) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

def get_primary_side(landmarks):
    left_vis  = landmarks[11].visibility
    right_vis = landmarks[12].visibility
    return "left" if left_vis > right_vis else "right"

def analyze_frame(landmarks):
    L = landmarks
    side = get_primary_side(L)

    if side == "left":
        shoulder_abduct = shoulder_abduction(L, "left")
        elbow_angle     = angle(lm(L,11), lm(L,13), lm(L,15))
        wrist_angle     = angle(lm(L,13), lm(L,15), lm(L,17))
    else:
        shoulder_abduct = shoulder_abduction(L, "right")
        elbow_angle     = angle(lm(L,12), lm(L,14), lm(L,16))
        wrist_angle     = angle(lm(L,14), lm(L,16), lm(L,18))

    return {
        "shoulder_abduct":  shoulder_abduct,
        "shoulder_side":    side,
        "elbow":            elbow_angle,
        "wrist":            wrist_angle,
        "hip_left":         angle(lm(L,11), lm(L,23), lm(L,25)),
        "hip_right":        angle(lm(L,12), lm(L,24), lm(L,26)),
        "knee_left":        angle(lm(L,23), lm(L,25), lm(L,27)),
        "knee_right":       angle(lm(L,24), lm(L,26), lm(L,28)),
        "ankle_left":       angle(lm(L,25), lm(L,27), lm(L,31)),
        "ankle_right":      angle(lm(L,26), lm(L,28), lm(L,32)),
        "spine_lean":       spine_angle(L),
        "trunk_rotation":   trunk_rotation(L),
    }

@app.route("/analyze", methods=["POST"])
def analyze():
    data      = request.json or {}
    video_url = data.get("video_url")
    if not video_url:
        return jsonify({"error": "video_url is required"}), 400

    try:
        r = requests.get(video_url, timeout=30)
        r.raise_for_status()
    except Exception as e:
        return jsonify({"error": f"Could not download video: {e}"}), 400

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(r.content)
        tmp_path = tmp.name

    try:
        cap          = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            return jsonify({"error": "Could not read video frames"}), 422

        start         = int(total_frames * 0.15)
        end           = int(total_frames * 0.85)
        frame_indices = [int(start + i * (end - start) / 31) for i in range(31)]

        all_frame_data = []
        failed_frames  = 0

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    failed_frames += 1
                    continue
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                if result.pose_landmarks:
                    all_frame_data.append(
                        analyze_frame(result.pose_landmarks.landmark)
                    )
                else:
                    failed_frames += 1

        cap.release()

        if not all_frame_data:
            return jsonify({
                "error":         "No pose detected in any frame",
                "failed_frames": failed_frames,
                "tip":           "Check camera angle — person should be fully visible",
            }), 422

        return jsonify({
            "frames_analyzed":        len(all_frame_data),
            "frames_failed":          failed_frames,
            "joint_angles_per_frame": all_frame_data,
        })

    finally:
        os.unlink(tmp_path)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
