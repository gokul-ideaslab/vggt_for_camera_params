import cv2
import os

video_path = "/home/gokul/nas/AI_Team/Datasets/scene3/camera_3_20260605_140222.mp4"
out_path = "calibration_images/winston/cam3"

os.makedirs(out_path, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Cannot open!")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot read")
        break

    if frame_count % 12 == 0:
        filename = os.path.join(out_path, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Wrote a frame {frame_count}")

    frame_count += 1

cap.release()