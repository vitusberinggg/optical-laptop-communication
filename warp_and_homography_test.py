import cv2
import numpy as np

from utilities.screen_alignment_functions import homography_matrix, warp_alignment

# -------------------------------
# Initialize global variables
# -------------------------------
window_name = "Homography Test"
warp_window = "Warped"
img_size = 400
dst_width = 800
dst_height = 500
# Initial trapezoid coordinates (TL, TR, BR, BL)
src_pts = np.array([
    [100, 150],
    [300, 130],
    [330, 300],
    [70, 300]
], dtype=np.float32)

dst_pts = np.array([
    [0, 0],
    [dst_width, 0],
    [dst_width, dst_height],
    [0, dst_height]
])

dragging_point = -1  # -1 = no point selected

# -------------------------------
# Mouse callback
# -------------------------------
def mouse_callback(event, x, y, flags, param):
    global dragging_point, src_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is near a corner
        for i, (px, py) in enumerate(src_pts):
            if abs(x - px) < 10 and abs(y - py) < 10:
                dragging_point = i
                break
    elif event == cv2.EVENT_MOUSEMOVE and dragging_point != -1:
        src_pts[dragging_point] = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_point = -1

# -------------------------------
# Main loop
# -------------------------------
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

while True:
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # Draw trapezoid
    cv2.polylines(canvas, [src_pts.astype(np.int32)], True, (255, 255, 255), 2)

    # Draw corners
    for i, (x, y) in enumerate(src_pts):
        cv2.circle(canvas, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(canvas, str(i), (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    H = homography_matrix(src_pts, dst_pts)

    # Warp and display
    warped = warp_alignment(canvas, H, dst_width, dst_height)
    cv2.imshow(window_name, canvas)
    cv2.imshow(warp_window, warped)

    key = cv2.waitKey(20) & 0xFF
    if key == 27:  # ESC to quit
        break

cv2.destroyAllWindows()
