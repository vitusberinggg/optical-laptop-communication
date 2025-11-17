# --- Imports ---
from recievers.webCamSim import VideoThreadedCapture
from recievers.color_utils import dominant_color, tracker  # 🔹 updated: import tracker
from utilities import detection_functions, screen_alignment_functions

import cv2
import time
import numpy as np

# --- Definitions ---
delimiter_duration = 0.5  # red duration
binary_duration = 0.3     # unused, just for reference
homography = None

# Match sender's screen size (from sender script)
sender_output_width = 2650
sender_output_height = 1440

# --- Setup capture ---
cap = VideoThreadedCapture(r"C:\my_projects\optical-laptop-communication\recievers\gandalf2.0.mp4")
# For live webcam test instead of video, use:
# cap = VideoThreadedCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera/video.")
    exit()

cv2.namedWindow("Receiver", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Receiver", 1920, 1200)

# Grab one initial frame so cap is "warmed up"
while True:
    ret, frame = cap.read()
    if ret:
        break
    time.sleep(0.01)

# --- ArUco setup (match sender) ---
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


# --- Main function ---
def receive_message():
    global homography

    bits = ""
    message = ""
    last_color = None
    waiting_for_sync = True
    decoding = False
    current_bit_colors = []  # DEBUG: store colors per bit

    print("Receiver started — waiting for GREEN to sync...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            continue

        # Flip horizontally to match your setup
        #frame = cv2.flip(frame, 1)

        # ---------- ArUco detection on the frame ----------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # If markers are low-contrast on screen, try thresholding instead:
        # th = cv2.adaptiveThreshold(
        #     gray, 255,
        #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #     cv2.THRESH_BINARY,
        #     11, 2
        # )
        # corners, ids, rejected = aruco_detector.detectMarkers(th)

        corners, ids, rejected = aruco_detector.detectMarkers(gray)

        rejected_count = 0 if rejected is None else len(rejected)
        print("ids:", None if ids is None else ids.flatten(), "rejected:", rejected_count)
        # --------------------------------------------------

        # Use warped ROI if homography is known, otherwise full frame
        if homography is not None:
            warped = cv2.warpPerspective(
                frame,
                homography,
                (sender_output_width, sender_output_height)
            )

            mask = screen_alignment_functions.create_mask(homography)
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

            # --- VISUALIZE MASK ---
            mask_color = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
            overlay = cv2.addWeighted(frame, 0.7, mask_color, 0.3, 0)
            cv2.imshow("Masked Overlay", overlay)

            roi = warped   # <-- use warped screen as the ROI
        else:
            roi = frame    # fallback while still syncing

        color = dominant_color(roi)

        # --- Visualization ---
        frame_with_roi = frame.copy()

        # Draw the pink polygon (homography ROI) if we have H
        if homography is not None:
            h, w = sender_output_height, sender_output_width
            corners_norm = np.array([
                [0, 0],         # top-left
                [w, 0],         # top-right
                [w, h],         # bottom-right
                [0, h]          # bottom-left
            ], dtype=np.float32).reshape(-1, 1, 2)

            projected_corners = cv2.perspectiveTransform(corners_norm, np.linalg.inv(homography))
            cv2.polylines(frame_with_roi, [np.int32(projected_corners)], True, (203, 192, 255), 3)

            cv2.polylines(frame_with_roi, [np.int32(projected_corners)], True, (0, 0, 255), 2)

        # ---- DRAW ARUCO INFO ON THE SAME FRAME ----
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame_with_roi, corners, ids)
            cv2.putText(frame_with_roi,
                        f"{len(ids)} ArUco marker(s) detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2)
        else:
            cv2.putText(frame_with_roi,
                        "No ArUco markers detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2)
        # -------------------------------------------

        cv2.imshow("Receiver", frame_with_roi)

        # --- SYNC ---
        if waiting_for_sync:
            if color == "green" and last_color != "green":
                print("Green detected — syncing...")
                homography = detection_functions.detect_aruco_marker_frame(frame)
                if homography is None:
                    print("[ARUCO] No homography found! Using full frame as ROI.")
                else:
                    print("[ARUCO] Homography OK. Will use warped ROI.")
                tracker.reset()
            elif color != "green" and last_color == "green":
                print("Green ended — starting decoding!")
                tracker.reset()
                waiting_for_sync = False
                decoding = True

        elif decoding:
            if color == "blue" and last_color != "blue":
                # end of bit → compute average color
                majority_color = tracker.end_bit()

                if majority_color == "white":
                    bits += "1"
                elif majority_color == "black":
                    bits += "0"

            elif color in ["white", "black"]:
                if homography is not None:
                    masked_roi = cv2.bitwise_and(frame, frame, mask=mask)
                    warped_roi = cv2.warpPerspective(masked_roi, homography, (sender_output_width, sender_output_height))
                    tracker.add_frame(warped_roi)
                else:
                    tracker.add_frame(frame)

            elif color == "red" and last_color != "red":
                while len(bits) >= 8:
                    byte = bits[:8]
                    bits = bits[8:]
                    try:
                        ch = chr(int(byte, 2))
                    except:
                        ch = '?'
                    message += ch
                    print(f"Received char: {ch}")
                if 0 < len(bits) < 8:
                    byte = bits.ljust(8, '0')
                    try:
                        ch = chr(int(byte, 2))
                    except:
                        ch = '?'
                    message += ch
                    print(f"Received char (padded): {ch}")
                bits = ""

        last_color = color
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # DEBUG: show any remaining colors and bits
    if current_bit_colors:
        print(f"Colors collected for last unfinished bit: {current_bit_colors}")
    if bits:
        print(f"Remaining bits not yet converted: {bits}")

    print("Final message:", message)
    cap.release()
    cv2.destroyAllWindows()


# --- Run ---
receive_message()
