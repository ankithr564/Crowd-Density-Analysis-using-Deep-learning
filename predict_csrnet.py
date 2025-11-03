import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from models.csrnet import CSRNet
import matplotlib.pyplot as plt

# ============================================================
# 📦 PATH CONFIGURATION
# ============================================================
base_path = r"C:\Users\Ankith R\OneDrive\Documents\mall_dataset\mall_dataset"
frames_dir = os.path.join(base_path, "frames")  # Folder containing .jpg frames
output_dir = base_path  # Save video here

model_path = r"C:\Users\Ankith R\OneDrive\Documents\mall_dataset\mall_dataset\checkpoint_best.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 🧠 LOAD CSRNet MODEL
# ============================================================
model = CSRNet()
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# ============================================================
# 🔄 IMAGE PREPROCESSING
# ============================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============================================================
# 🎥 GENERATE PREDICTION VIDEO
# ============================================================
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
frame_height, frame_width = first_frame.shape[:2]

output_path = os.path.join(output_dir, "predicted_density.avi")
out_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_width, frame_height))

for frame_name in frame_files:
    frame_path = os.path.join(frames_dir, frame_name)
    img = cv2.imread(frame_path)
    if img is None:
        print(f"⚠️ Skipping unreadable file: {frame_name}")
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_img = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        density_map = model(input_img).cpu().numpy()[0, 0, :, :]

    count = np.sum(density_map)

    # Normalize for visualization
    density_map = density_map / (density_map.max() + 1e-5)
    density_map_colored = cv2.applyColorMap((density_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    density_map_colored = cv2.resize(density_map_colored, (frame_width, frame_height))

    overlay = cv2.addWeighted(img, 0.6, density_map_colored, 0.4, 0)
    cv2.putText(overlay, f"Predicted Count: {int(count)}", (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    out_video.write(overlay)
    print(f"Processed {frame_name} → Count: {int(count)}")

out_video.release()
print(f"✅ Video saved at: {output_path}")
