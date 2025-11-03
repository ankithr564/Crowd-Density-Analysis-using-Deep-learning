import cv2

# Load pre-trained Haar Cascade for face/head detection
# You can also use 'haarcascade_frontalface_default.xml'
# For heads, sometimes people use 'haarcascade_fullbody.xml' or custom models
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
head_cascade = cv2.CascadeClassifier(cascade_path)

# Start webcam
cap = cv2.VideoCapture(0)  # 0 = default laptop camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster detection
    frame_resized = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # Detect heads/faces
    heads = head_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw red dots on each head
    for (x, y, w, h) in heads:
        center = (x + w//2, y + h//2)
        cv2.circle(frame_resized, center, 5, (0, 0, 255), -1)  # red dot

    # Display total head count
    cv2.putText(frame_resized, f"Head Count: {len(heads)}", (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

    # Show video
    cv2.imshow("Head Counting", frame_resized)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
a="djfhdbsfsdfhbdfjhjdbffadf"
b='whdabsdjdhbjsdhbsd'
