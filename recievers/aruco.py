import cv2
import numpy as np

# --- Settings ---
source =r"C:\Users\eanpaln\Videos\Screen Recordings\Recordinggg.mp4"  # 0 for webcam, or a video file path
aruco_dict_type = cv2.aruco.DICT_4X4_50
corner_marker_ids = [0, 1, 2, 3]  # IDs of the four corner markers

# --- Initialize capture ---
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print("Error: Cannot open source", source)
    exit()

# --- ArUco setup for OpenCV >=4.7 ---
aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, params)

print("Detecting ArUco markers. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Detect markers ---
    corners, ids, _ = detector.detectMarkers(frame)

    display = frame.copy()

    if ids is not None and len(ids) > 0:
        ids_flat = ids.flatten()
        # Draw all detected markers
        cv2.aruco.drawDetectedMarkers(display, corners, ids)

        # Highlight the corner markers
        corner_pts = []
        for idx, marker_id in enumerate(ids_flat):
            if marker_id in corner_marker_ids:
                c = corners[idx][0]  # marker corners
                center = c.mean(axis=0).astype(int)
                corner_pts.append(center)
                cv2.circle(display, tuple(center), 10, (0, 255, 255), -1)
                cv2.putText(display, f"ID:{marker_id}", tuple(center + [10,10]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        if len(corner_pts) == 4:
            # Draw polygon connecting corner markers
            pts = np.array(corner_pts, dtype=np.int32)
            cv2.polylines(display, [pts], isClosed=True, color=(0,255,0), thickness=3)
    else:
        cv2.putText(display, "No markers detected", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Aruco Detection", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
