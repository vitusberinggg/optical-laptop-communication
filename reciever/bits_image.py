import cv2
import os

print("Current working directory:", os.getcwd())

cap = cv2.VideoCapture(r"C:\my_projects\optical-laptop-communication\reciever\screen_r.mp4")

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # fallback if OpenCV can’t read it
delay = int(1000 / fps)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Feed", frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
