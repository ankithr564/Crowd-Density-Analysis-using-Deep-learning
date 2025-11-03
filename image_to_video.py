# # jpg_to_count_video.py
# import os
# import cv2
# import scipy.io as sio
# from tqdm import tqdm

# # Paths
# data_path = r"C:\Users\Ankith R\OneDrive\Documents\mall_dataset\mall_dataset"
# frames_dir = os.path.join(data_path, "frames")
# gt_path = os.path.join(data_path, "mall_gt.mat")
# output_video_path = os.path.join(data_path, "mall_count_video.mp4")

# fps = 10
# frame_size = (512, 512)

# # Load GT
# gt = sio.loadmat(gt_path)
# annotations = gt["frame"]

# # Video writer
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_size[0], frame_size[1]))

# frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg") or f.endswith(".png")])
# for i, fname in enumerate(tqdm(frames)):
#     frame_path = os.path.join(frames_dir, fname)
#     img = cv2.imread(frame_path)
#     img = cv2.resize(img, frame_size)
#     try:
#         points = annotations[0][i][0][0][0]
#     except:
#         points = []

#     # Draw red dots
#     for p in points:
#         x = int(p[0] * frame_size[0] / img.shape[1])
#         y = int(p[1] * frame_size[1] / img.shape[0])
#         cv2.circle(img, (x, y), radius=4, color=(0,0,255), thickness=-1)

#     # Display count
#     count = len(points)
#     cv2.putText(img, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

#     # Write to video
#     out.write(img)

# out.release()
# print("✅ Video saved to:", output_video_path)

import os
import cv2
import scipy.io as sio
from tqdm import tqdm

# -----------------------------
# Paths
# -----------------------------
data_path = r"C:\Users\Ankith R\OneDrive\Documents\mall_dataset\mall_dataset"
frames_dir = os.path.join(data_path, "frames")
gt_path = os.path.join(data_path, "mall_gt.mat")
output_video_path = os.path.join(data_path, "mall_count_video.mp4")

fps = 2.5
frame_size = (512, 512)  # width, height

# -----------------------------
# Load ground truth
# -----------------------------
gt = sio.loadmat(gt_path)
annotations = gt["frame"]

# -----------------------------
# Prepare video writer
# -----------------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg") or f.endswith(".png")])

# -----------------------------
# Process frames
# -----------------------------
for i, fname in enumerate(tqdm(frames)):
    frame_path = os.path.join(frames_dir, fname)
    img = cv2.imread(frame_path)
    orig_h, orig_w = img.shape[:2]
    img = cv2.resize(img, frame_size)
    target_w, target_h = frame_size

    # Get head positions from GT
    try:
        points = annotations[0][i][0][0][0]  # nested structure
    except:
        points = []

    # Draw red dots at scaled coordinates
    for p in points:
        x = int(p[0] * target_w / orig_w)
        y = int(p[1] * target_h / orig_h)
        cv2.circle(img, (x, y), radius=2, color=(0,0,250), thickness=1)

    # Display count
    count = len(points)
    cv2.putText(img, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    # Write to video
    out.write(img)

out.release()
print("✅ Video saved to:", output_video_path)

