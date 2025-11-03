import urllib.request

url = "https://github.com/SkalskiP/HeadDetectionYOLO/releases/download/v1.0/yolov8n-head.pt"
output = "yolov8n-head.pt"

try:
    print("📦 Downloading YOLOv8 Head Model...")
    urllib.request.urlretrieve(url, output)
    print(f"✅ Download completed! Saved as {output}")
except Exception as e:
    print("❌ Download failed:", e)
