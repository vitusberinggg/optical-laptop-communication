import cv2

cap = cv2.VideoCapture(r"C:\my_projects\optical-laptop-communication\recievers\gandalf.mp4")

if not cap.isOpened():
    print("Cannot open video")
    exit()

print("--- OpenCV thinks the video resolution is ---")
print("Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("RAW Video", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
