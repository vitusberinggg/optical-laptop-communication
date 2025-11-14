import cv2
import numpy as np

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

cap = cv2.VideoCapture(r"C:\Users\eanpaln\Videos\Screen Recordings\Screen Recording 2025-11-14 134827.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_small = cv2.resize(frame, (1280, 720))  # resize for detection
    corners, ids, _ = detector.detectMarkers(frame_small)
    
    display = frame_small.copy()
    
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(display, corners, ids)
        print("Detected IDs:", ids.flatten())
    else:
        print("No markers detected")

    cv2.imshow("Debug ArUco", display)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
