import cv2
import numpy as np

# --- Settings ---
screen_width = 800
screen_height = 600
marker_size = 120
marker_ids = [0, 1, 2, 3]  # IDs for top-left, top-right, bottom-right, bottom-left
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# --- Create markers ---
def create_marker(id, size=marker_size):
    marker = np.zeros((size, size), dtype=np.uint8)
    cv2.aruco.generateImageMarker(aruco_dict, id, size, marker, 1)  # new function
    return cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

markers = [create_marker(i) for i in marker_ids]

# --- Create blank screen and place markers ---
frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
positions = [
    (0, 0),  # top-left
    (screen_width - marker_size, 0),  # top-right
    (screen_width - marker_size, screen_height - marker_size),  # bottom-right
    (0, screen_height - marker_size)  # bottom-left
]

for marker, (x, y) in zip(markers, positions):
    frame[y:y+marker_size, x:x+marker_size] = marker

# --- Display ---
cv2.namedWindow("Sender", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Sender", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Press 'q' to quit")
while True:
    cv2.imshow("Sender", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
