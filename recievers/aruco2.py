import cv2
import numpy as np

# --- Settings ---
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

# Example: list of image file paths
image_files = [
    "frame1.png",
    "frame2.png",
    "frame3.png"
]

for file in image_files:
    frame = cv2.imread(file)
    if frame is None:
        print(f"Could not read {file}")
        continue

    corners, ids, _ = detector.detectMarkers(frame)

    display = frame.copy()
    if ids is not None and len(ids) >= 4:
        cv2.aruco.drawDetectedMarkers(display, corners, ids)

        # get center points of each marker
        centers = [c[0].mean(axis=0) for c in corners]

        # sort top-left, top-right, bottom-right, bottom-left
        centers_sorted = sorted(centers, key=lambda x: (x[1], x[0]))  # sort by y, then x
        top_two = sorted(centers_sorted[:2], key=lambda x: x[0])
        bottom_two = sorted(centers_sorted[2:4], key=lambda x: x[0])
        ordered = [top_two[0], top_two[1], bottom_two[1], bottom_two[0]]

        pts = np.array(ordered, dtype=np.int32)
        cv2.polylines(display, [pts], isClosed=True, color=(0, 255, 255), thickness=3)
        cv2.putText(display, "Square drawn!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    else:
        cv2.putText(display, "Markers not detected", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Receiver", display)
    key = cv2.waitKey(0)  # wait until a key is pressed
    if key == ord('q'):
        break

cv2.destroyAllWindows()
