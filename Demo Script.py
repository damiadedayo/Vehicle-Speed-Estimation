import cv2
import numpy as np
import math
import pickle
import os
import sys

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict

# =============================================================
# DEMO — Calibration-Free Vehicle Speed Estimation (v3)
# Damilola Adedayo | 23021899 | Staffordshire University
#
# Usage:
#   python demo.py <video_path> [annotation_path]
#
# annotation_path is optional. If provided, ground truth speed
# is read and error is reported. If omitted, only the predicted
# speed is printed.
#
# Example (with ground truth):
#   python demo.py "VS13 Dataset/Mazda3/clip_001.mp4" "VS13 Dataset/Mazda3/clip_001.txt"
#
# Example (no ground truth):
#   python demo.py "myvideo.mp4"
# =============================================================

# =============================================================
# PATHS — update MODEL_PATH and PKL_PATH to match your machine
# =============================================================

MODEL_PATH = "yolov8n.pt"
PKL_PATH   = "xgb_1s_speed_model_final_v3.pkl"

# =============================================================
# CONFIG
# =============================================================

CONF_THRESHOLD  = 0.4
IOU_THRESHOLD   = 0.5
VEHICLE_CLASSES = [2, 3, 5, 7]   # car, motorbike, bus, truck
WINDOW_BEFORE   = 1.0             # seconds before exit frame

# =============================================================
# PARSE ARGUMENTS
# =============================================================

if len(sys.argv) < 2:
    print("\nUsage: python demo.py <video_path> [annotation_path]")
    print("  annotation_path is optional — provide to compare against ground truth.")
    sys.exit(1)

VIDEO_PATH = sys.argv[1]
ANN_PATH   = sys.argv[2] if len(sys.argv) >= 3 else None

if not os.path.exists(VIDEO_PATH):
    print(f"\nError: video file not found: {VIDEO_PATH}")
    sys.exit(1)

if ANN_PATH and not os.path.exists(ANN_PATH):
    print(f"\nError: annotation file not found: {ANN_PATH}")
    sys.exit(1)

# =============================================================
# LOAD MODELS
# =============================================================

print("\nLoading models...")

yolo = YOLO(MODEL_PATH)

with open(PKL_PATH, "rb") as f:
    xgb_model = pickle.load(f)

print("  YOLOv8n      : loaded")
print("  XGBoost v3   : loaded")

# =============================================================
# READ ANNOTATION (optional)
# =============================================================

ground_truth_speed = None

if ANN_PATH:
    with open(ANN_PATH, "r") as f:
        ground_truth_speed = float(f.read().split()[0])
    print(f"\nGround truth speed : {ground_truth_speed:.1f} km/h")

print(f"\nProcessing video   : {os.path.basename(VIDEO_PATH)}")
print("Running detection and tracking — single pass, exit anchored...\n")

# =============================================================
# VIDEO SETUP
# =============================================================

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

tracker = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.2)

trajectories = defaultdict(list)
heights      = defaultdict(list)
widths       = defaultdict(list)

frame_idx = 0

# =============================================================
# SINGLE FULL-VIDEO PASS — detection + tracking
# =============================================================

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    frame = cv2.resize(frame, (1280, 720))

    results = yolo(
        frame,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        classes=VEHICLE_CLASSES,
        verbose=False
    )

    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            w = x2 - x1
            h = y2 - y1
            detections.append(([x1, y1, w, h], conf, "vehicle"))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:

        if not track.is_confirmed():
            continue
        if track.time_since_update > 0:
            continue

        track_id = track.track_id
        l, t, r_coord, b = track.to_ltrb()

        x1 = int(l);  y1 = int(t)
        x2 = int(r_coord); y2 = int(b)

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        trajectories[track_id].append((frame_idx, cx, cy))
        heights[track_id].append((frame_idx, y2 - y1))
        widths[track_id].append((frame_idx, x2 - x1))

cap.release()

# =============================================================
# SELECT LONGEST TRACK (main vehicle heuristic)
# =============================================================

if not trajectories:
    print("No tracks found in video. Exiting.")
    sys.exit(1)

main_track = max(trajectories, key=lambda k: len(trajectories[k]))

all_points = trajectories[main_track]
all_hs     = heights[main_track]
all_ws     = widths[main_track]

if len(all_points) < 5:
    print("Track too short for reliable estimation. Exiting.")
    sys.exit(1)

# =============================================================
# EXIT ANCHOR — 1.0s window ending at last confirmed frame
# =============================================================

exit_frame   = all_points[-1][0]
window_start = exit_frame - int(WINDOW_BEFORE * fps)

points = [(cx, cy) for fi, cx, cy in all_points if fi >= window_start]
hs     = [h        for fi, h      in all_hs     if fi >= window_start]
ws_w   = [w        for fi, w      in all_ws     if fi >= window_start]

print(f"Track length       : {len(all_points)} frames  ({len(all_points)/fps:.2f}s)")
print(f"Exit frame         : {exit_frame}")
print(f"Extraction window  : frames {window_start}–{exit_frame}  ({len(points)} points)")

if len(points) < 5:
    print("Not enough points in window for estimation. Exiting.")
    sys.exit(1)

# =============================================================
# FEATURE EXTRACTION — v3 feature set (23 features)
# =============================================================

dxs        = []
dys        = []
euclideans = []
speeds_px  = []

for i in range(1, len(points)):
    x1c, y1c = points[i - 1]
    x2c, y2c = points[i]
    dx   = abs(x2c - x1c)
    dy   = abs(y2c - y1c)
    dxs.append(dx)
    dys.append(dy)
    dist = math.sqrt(dx ** 2 + dy ** 2)
    euclideans.append(dist)
    speeds_px.append(dist * fps)

# Motion
mean_dy          = np.mean(dys)
max_dy           = np.max(dys)
std_dy           = np.std(dys)
mean_dx          = np.mean(dxs)
max_dx           = np.max(dxs)
std_dx           = np.std(dxs)
std_euclidean    = np.std(euclideans)
dx_dy_ratio      = mean_dx / mean_dy if mean_dy != 0 else 0
mean_pixel_speed = np.mean(speeds_px)

# Height
height_deltas         = [hs[i] - hs[i - 1] for i in range(1, len(hs))]
mean_height           = np.mean(hs)
max_height            = np.max(hs)
height_growth         = (hs[-1] - hs[0]) / hs[0] if hs[0] != 0 else 0
height_change_rate    = np.mean(height_deltas)
std_height_delta      = np.std(height_deltas)
# height_expansion_rate — PRUNED, not included

# Width
width_deltas         = [ws_w[i] - ws_w[i - 1] for i in range(1, len(ws_w))]
mean_width           = np.mean(ws_w)
width_growth         = (ws_w[-1] - ws_w[0]) / ws_w[0] if ws_w[0] != 0 else 0
# width_change_rate — PRUNED, not included
std_width_delta      = np.std(width_deltas)
max_width_delta      = np.max(width_deltas)
width_expansion_rate = (ws_w[-1] - ws_w[0]) / WINDOW_BEFORE

# Combined / ratio
aspect_ratio             = mean_width / mean_height if mean_height != 0 else 0
initial_ar               = ws_w[0]  / hs[0]  if hs[0]  != 0 else 0
final_ar                 = ws_w[-1] / hs[-1] if hs[-1] != 0 else 0
aspect_ratio_change_rate = (final_ar - initial_ar) / WINDOW_BEFORE
TTC                      = mean_width / width_expansion_rate if width_expansion_rate != 0 else 0

# Derived (computed at inference, not extracted)
width_growth_norm = width_growth / mean_width if mean_width != 0 else 0

# track_length  — PRUNED
# vehicle_class — PRUNED

# =============================================================
# ASSEMBLE FEATURE VECTOR
# Column order must match training exactly (train_final_with_heldout_v2.py)
# =============================================================

features = [[
    mean_dy, max_dy, std_dy,
    mean_dx, max_dx, std_dx,
    std_euclidean, dx_dy_ratio, mean_pixel_speed,
    mean_height, max_height, height_growth, height_change_rate,
    mean_width, width_growth, aspect_ratio,
    std_height_delta,
    width_expansion_rate, std_width_delta, max_width_delta,
    TTC, aspect_ratio_change_rate,
    width_growth_norm
]]

# =============================================================
# PREDICT
# =============================================================

predicted_speed = float(xgb_model.predict(features)[0])

print(f"\n{'='*40}")
print(f"  PREDICTED SPEED  :  {predicted_speed:.2f} km/h")

if ground_truth_speed is not None:
    error = abs(predicted_speed - ground_truth_speed)
    print(f"  GROUND TRUTH     :  {ground_truth_speed:.1f} km/h")
    print(f"  ABSOLUTE ERROR   :  {error:.2f} km/h")

print(f"{'='*40}\n")