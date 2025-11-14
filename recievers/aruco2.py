import cv2
import numpy as np

# --- Parameters ---
video_path = r"C:\Users\eanpaln\Videos\Screen Recordings\Screen Recording 2025-11-14 134827.mp4"  # Change to your video path
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

# Define sender screen output size (used for homography)
sender_output_width = 800
sender_output_height = 600

def detect_aruco_marker_frame(frame):
    """
    Detects four ArUco markers in a frame and computes the homography.
    Draws detected markers and a polygon connecting them.
    """
    corners, ids, _ = detector.detectMarkers(frame)
    display = frame.copy()

    if ids is None or len(ids) < 4:
        cv2.putText(display, "Markers not detected", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return display, None

    ids = ids.flatten()
    # Sort markers by ID 0,1,2,3
    sorted_corners = [None]*4
    for c, id_ in zip(corners, ids):
        if id_ < 4:
            sorted_corners[id_] = c[0]

    if any(c is None for c in sorted_corners):
        cv2.putText(display, "Not all markers detected", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return display, None

    # Draw markers
    cv2.aruco.drawDetectedMarkers(display, corners, ids)

    # Draw polygon connecting the markers
    pts = np.array([sorted_corners[0], sorted_corners[1], sorted_corners[3], sorted_corners[2]], dtype=np.int32)
    cv2.polylines(display, [pts], isClosed=True, color=(0,255,255), thickness=3)

    # Compute homography
    source = np.array([sorted_corners[0], sorted_corners[1], sorted_corners[3], sorted_corners[2]], dtype=np.float32)
    destination = np.array([[0,0],[sender_output_width,0],[sender_output_width,sender_output_height],[0,sender_output_height]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(source, destination)

    cv2.putText(display, "Markers detected!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    return display, homography

# --- Main loop ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video reached.")
        break

    display, homography = detect_aruco_marker_frame(frame)
    if homography is not None:
        print("Homography matrix:\n", homography)

    cv2.imshow("Aruco Detection", display)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
