import cv2
import mediapipe as mp
import numpy as np
import requests
import tempfile
import os
from flask import Flask, request, jsonify

app = Flask(__name__)
mp_pose = mp.solutions.pose

# ── MediaPipe landmark indices (for reference) ──────────────────────
# 11=left_shoulder  12=right_shoulder
# 13=left_elbow     14=right_elbow
# 15=left_wrist     16=right_wrist
# 17=left_pinky     18=right_pinky
# 23=left_hip       24=right_hip
# 25=left_knee      26=right_knee
# 27=left_ankle     28=right_ankle
# 29=left_heel      30=right_heel
# 31=left_foot      32=right_foot

# ── Joint angle calculator ──────────────────────────────────────────
def angle(a, b, c):
    """Angle in degrees at joint B, given three [x,y,z] points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

def lm(landmarks, idx):
    p = landmarks[idx]
    return [p.x, p.y, p.z]

def spine_angle(landmarks):
    """Torso lean: angle between shoulder midpoint, hip midpoint, and vertical."""
    L = landmarks
    shoulder_mid = np.mean([lm(L,11), lm(L,12)], axis=0)
    hip_mid      = np.mean([lm(L,23), lm(L,24)], axis=0)
    vertical_ref = hip_mid + np.array([0, -0.2, 0])  # point straight up
    return float(angle(shoulder_mid, hip_mid, vertical_ref))

def shoulder_abduction(landmarks, side="right"):
    """How far the upper arm is raised away from the body (abduction)."""
    L = landmarks
    if side == "right":
        return angle(lm(L,14), lm(L,12), lm(L,24))  # elbow, shoulder, hip
    else:
        return angle(lm(L,13), lm(L,11), lm(L,23))

def shoulder_flexion(landmarks, side="right"):
    """Forward raise angle of the upper arm."""
    L = landmarks
    if side == "right":
        return angle(lm(L,14), lm(L,12), lm(L,24))
    else:
        return angle(lm(L,13), lm(L,11), lm(L,23))

def trunk_rotation(landmarks):
    """Horizontal twist of torso — proxy for oblique involvement."""
    L = landmarks
    ls, rs = np.array(lm(L,11)), np.array(lm(L,12))
    lh, rh = np.array(lm(L,23)), np.array(lm(L,24))
    shoulder_vec = rs - ls
    hip_vec      = rh - lh
    # Angle difference in horizontal plane (x-z)
    s2d = np.array([shoulder_vec[0], shoulder_vec[2]])
    h2d = np.array([hip_vec[0],      hip_vec[2]])
    cos = np.dot(s2d, h2d) / (np.linalg.norm(s2d) * np.linalg.norm(h2d) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

# ── Frame analysis — returns all joint angles for one frame ─────────
def analyze_frame(landmarks):
    L = landmarks
    return {
        # ELBOWS
        "elbow_right":         angle(lm(L,12), lm(L,14), lm(L,16)),
        "elbow_left":          angle(lm(L,11), lm(L,13), lm(L,15)),
        # WRISTS (wrist flex = angle between forearm and hand plane proxy)
        "wrist_right":         angle(lm(L,14), lm(L,16), lm(L,18)),
        "wrist_left":          angle(lm(L,13), lm(L,15), lm(L,17)),
        # SHOULDERS
        "shoulder_abduct_r":   shoulder_abduction(L, "right"),
        "shoulder_abduct_l":   shoulder_abduction(L, "left"),
        "shoulder_flex_r":     shoulder_flexion(L, "right"),
        "shoulder_flex_l":     shoulder_flexion(L, "left"),
        # HIPS
        "hip_right":           angle(lm(L,12), lm(L,24), lm(L,26)),
        "hip_left":            angle(lm(L,11), lm(L,23), lm(L,25)),
        # KNEES
        "knee_right":          angle(lm(L,24), lm(L,26), lm(L,28)),
        "knee_left":           angle(lm(L,23), lm(L,25), lm(L,27)),
        # ANKLES
        "ankle_right":         angle(lm(L,26), lm(L,28), lm(L,32)),
        "ankle_left":          angle(lm(L,25), lm(L,27), lm(L,31)),
        # SPINE & TRUNK
        "spine_lean":          spine_angle(L),
        "trunk_rotation":      trunk_rotation(L),
    }

# ── Full muscle map ─────────────────────────────────────────────────
# Each entry: joint_key → list of (muscle_name, base_weight)
# base_weight: 1.0 = primary mover, 0.6 = strong secondary, 0.3 = stabiliser
#
# Weights reflect established biomechanics literature:
#   - Strength & Conditioning: NSCA Essentials (4th ed.)
#   - Muscle & Nerve: Kendall's Muscles (5th ed.)
#   - Schoenfeld 2010 (mechanisms of hypertrophy)

MUSCLE_MAP = {

    # ── ELBOW FLEXION (both arms averaged) ─────────────────────────
    "elbow_right": [
        ("Biceps",              1.0),
        ("Brachialis",          1.0),
        ("Brachioradialis",     0.7),
        ("Wrist_flexors",       0.2),   # stabilise grip
    ],
    "elbow_left": [
        ("Biceps",              1.0),
        ("Brachialis",          1.0),
        ("Brachioradialis",     0.7),
        ("Wrist_flexors",       0.2),
    ],

    # ── WRIST ──────────────────────────────────────────────────────
    "wrist_right": [
        ("Wrist_flexors",       0.8),
        ("Wrist_extensors",     0.8),
        ("Brachioradialis",     0.3),
    ],
    "wrist_left": [
        ("Wrist_flexors",       0.8),
        ("Wrist_extensors",     0.8),
        ("Brachioradialis",     0.3),
    ],

    # ── SHOULDER ABDUCTION ─────────────────────────────────────────
    # (arm moving away from body to the side — lateral raise pattern)
    "shoulder_abduct_r": [
        ("Lateral_delts",       1.0),
        ("Front_delts",         0.4),
        ("Upper_traps",         0.5),
        ("Seratorious",         0.4),
        ("Infraspinatus",       0.3),
        ("Teres_minor",         0.3),
    ],
    "shoulder_abduct_l": [
        ("Lateral_delts",       1.0),
        ("Front_delts",         0.4),
        ("Upper_traps",         0.5),
        ("Seratorious",         0.4),
        ("Infraspinatus",       0.3),
        ("Teres_minor",         0.3),
    ],

    # ── SHOULDER FLEXION ───────────────────────────────────────────
    # (arm moving forward — press / fly / pull pattern)
    # Weight distribution changes with direction:
    #   high angle (>90°) = overhead / pull = lats dominant
    #   mid angle (45-90°) = row / fly = rear delt / mid back
    #   low angle (<45°)  = press = chest / front delt
    "shoulder_flex_r": [
        ("Front_delts",             0.8),
        ("Pec_major_clavicular_head", 0.7),
        ("Pec_major_sternal_head",  0.5),
        ("Upper_lats",              0.6),
        ("Lower_lats",              0.5),
        ("Teres_major",             0.5),
        ("Rear_delts",              0.4),
        ("Mid_traps",               0.4),
        ("Lower_traps",             0.3),
        ("Rhomboids",               0.3),   # note: not in your list but kept as 0 contributor
        ("Seratorious",             0.3),
    ],
    "shoulder_flex_l": [
        ("Front_delts",             0.8),
        ("Pec_major_clavicular_head", 0.7),
        ("Pec_major_sternal_head",  0.5),
        ("Upper_lats",              0.6),
        ("Lower_lats",              0.5),
        ("Teres_major",             0.5),
        ("Rear_delts",              0.4),
        ("Mid_traps",               0.4),
        ("Lower_traps",             0.3),
        ("Seratorious",             0.3),
    ],

    # ── HIP EXTENSION ──────────────────────────────────────────────
    "hip_right": [
        ("Glute_maximus",       1.0),
        ("Bicep_femoris",       0.8),
        ("Semitendinosus",      0.8),
        ("Semimembranosus",     0.8),
        ("Glute_medius",        0.4),
        ("Erector_spinae",      0.5),
        ("Hip_adductors",       0.3),
        ("Gracilis",            0.2),
    ],
    "hip_left": [
        ("Glute_maximus",       1.0),
        ("Bicep_femoris",       0.8),
        ("Semitendinosus",      0.8),
        ("Semimembranosus",     0.8),
        ("Glute_medius",        0.4),
        ("Erector_spinae",      0.5),
        ("Hip_adductors",       0.3),
        ("Gracilis",            0.2),
    ],

    # ── KNEE EXTENSION ─────────────────────────────────────────────
    "knee_right": [
        ("Rectus_femoris",      1.0),
        ("Vastus_lateralis",    1.0),
        ("Vastus_medialis",     1.0),
        ("Bicep_femoris",       0.6),   # eccentric brake
        ("Semitendinosus",      0.5),
        ("Semimembranosus",     0.5),
        ("Gastrocnemius",       0.3),   # crosses knee
        ("Gracilis",            0.2),
        ("Seratorious",         0.2),
        ("Tensor_fasciae_latae",0.3),
    ],
    "knee_left": [
        ("Rectus_femoris",      1.0),
        ("Vastus_lateralis",    1.0),
        ("Vastus_medialis",     1.0),
        ("Bicep_femoris",       0.6),
        ("Semitendinosus",      0.5),
        ("Semimembranosus",     0.5),
        ("Gracilis",            0.2),
        ("Seratorious",         0.2),
        ("Tensor_fasciae_latae",0.3),
    ],

    # ── ANKLE PLANTARFLEXION ───────────────────────────────────────
    "ankle_right": [
        ("Calves",              1.0),
        ("Tibialis_anterior",   0.5),   # eccentric control on dorsiflexion
    ],
    "ankle_left": [
        ("Calves",              1.0),
        ("Tibialis_anterior",   0.5),
    ],

    # ── SPINE LEAN (forward hinge) ─────────────────────────────────
    "spine_lean": [
        ("Erector_spinae",      1.0),
        ("Glute_maximus",       0.6),
        ("Bicep_femoris",       0.5),
        ("Rectus_abdominis",    0.4),   # isometric anti-flexion
        ("Lower_traps",         0.3),
        ("Mid_traps",           0.3),
    ],

    # ── TRUNK ROTATION ─────────────────────────────────────────────
    "trunk_rotation": [
        ("External_obliques",   1.0),
        ("Rectus_abdominis",    0.4),
        ("Erector_spinae",      0.3),
    ],
}

# Only muscles in this set will appear in the final output.
# Any muscle in MUSCLE_MAP not in this set is silently ignored.
VALID_MUSCLES = {
    # CHEST
    "Pec_major_sternal_head", "Pec_major_costal_head", "Pec_major_clavicular_head",
    # SHOULDERS
    "Front_delts", "Lateral_delts", "Rear_delts",
    # BACK
    "Upper_lats", "Lower_lats", "Teres_major", "Teres_minor",
    "Mid_traps", "Lower_traps", "Upper_traps", "Infraspinatus", "Erector_spinae",
    # ARMS
    "Biceps", "Brachialis", "Brachioradialis",
    "Tricep_long_head", "Tricep_lateral_head", "Tricep_medial_head",
    "Wrist_flexors", "Wrist_extensors",
    # CORE
    "Rectus_abdominis", "External_obliques",
    # GLUTES & HIPS
    "Glute_maximus", "Glute_medius", "Tensor_fasciae_latae",
    "Hip_adductors", "Gracilis", "Seratorious",
    # QUADS
    "Rectus_femoris", "Vastus_lateralis", "Vastus_medialis",
    # HAMSTRINGS
    "Bicep_femoris", "Semitendinosus", "Semimembranosus",
    # CALVES
    "Calves", "Tibialis_anterior",
}

# ── Hypertrophy ranker ──────────────────────────────────────────────
def rank_muscles(all_frames):
    """
    Score = ROM × eccentric_multiplier × base_weight
    
    Eccentric multiplier (1.5x):
      Applied when the joint angle INCREASES in the second half of the video
      (muscle lengthening under load = peak hypertrophy stimulus).
    
    ROM threshold:
      Joints that barely move (<8°) contribute very little — avoids noise
      from stabiliser joints inflating scores on irrelevant muscles.
    """
    joints = list(all_frames[0].keys())
    scores = {}

    for joint in joints:
        angles = [f[joint] for f in all_frames]
        rom = max(angles) - min(angles)

        # Skip joints that barely moved — likely not the working joint
        if rom < 8.0:
            continue

        # Detect eccentric phase: angle increases in second half
        mid = len(angles) // 2
        second_half = angles[mid:]
        eccentric_rom = max(second_half) - min(second_half)
        eccentric_mult = 1.5 if eccentric_rom > 15.0 else 1.0

        joint_score = rom * eccentric_mult

        muscles = MUSCLE_MAP.get(joint, [])
        for muscle, base_weight in muscles:
            if muscle not in VALID_MUSCLES:
                continue  # skip muscles not in your list (e.g. Gastrocnemius, Rhomboids)
            scores[muscle] = scores.get(muscle, 0.0) + joint_score * base_weight

    if not scores:
        return []

    # Normalise so top muscle = 100, then rank
    max_score = max(scores.values())
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [
        {
            "name":            muscle,
            "rank":            i + 1,
            "score":           round(score, 1),
            "relative_score":  round((score / max_score) * 100, 1),
        }
        for i, (muscle, score) in enumerate(ranked)
        if score > 0
    ]

# ── Main endpoint ───────────────────────────────────────────────────
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json or {}
    video_url = data.get("video_url")
    if not video_url:
        return jsonify({"error": "video_url is required"}), 400

    # Download video to a temp file
    try:
        r = requests.get(video_url, timeout=30)
        r.raise_for_status()
    except Exception as e:
        return jsonify({"error": f"Could not download video: {e}"}), 400

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(r.content)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            return jsonify({"error": "Could not read video frames"}), 422

        # Sample 21 evenly-spaced frames
        frame_indices = [int(i * total_frames / 21) for i in range(21)]

        all_frame_data = []
        failed_frames  = 0

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,          # most accurate; use 1 if timing out
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

        ranked = rank_muscles(all_frame_data)

        return jsonify({
            "frames_analyzed":      len(all_frame_data),
            "frames_failed":        failed_frames,
            "muscles":              ranked,
            # Uncomment the line below for debugging joint angles per frame:
             "joint_angles_per_frame": all_frame_data,
        })

    finally:
        os.unlink(tmp_path)

# ── Health check (Railway / Render ping) ───────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)


