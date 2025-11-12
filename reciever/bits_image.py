import cv2
import time
import os

print("Current working directory:", os.getcwd())

cap = cv2.VideoCapture(r"C:\my_projects\optical-laptop-communication\reciever\screen_r.mp4")

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # fallback if OpenCV can’t read it

frame_time = 1.0 / fps  # target time per frame in seconds

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Feed", frame)

    # calculate how long frame processing took
    elapsed = time.time() - start_time
    sleep_time = frame_time - elapsed

    # only sleep if we’re ahead of schedule
    if sleep_time > 0:
        time.sleep(sleep_time)

    # quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
