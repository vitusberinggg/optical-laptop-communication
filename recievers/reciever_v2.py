# --- Imports ---
import cv2
import numpy as np
from utilities.detect_end_frame import detect_end_frame
from utilities.bits_to_message import bits_to_message
from utilities.detect_start_frame import detect_start_frame
from utilities.decode_bits_with_blue import decode_bits_with_blue
from utilities.global_definitions import frame_duration

# --- Detect sender screen using ArUco markers ---
def detect_screen(frame):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    if ids is None or len(ids) < 4:
        return None  # not enough markers detected

    marker_dict = {id[0]: corner for id, corner in zip(ids, corners)}

    try:
        pts_src = np.array([
            marker_dict[0][0][0],  # top-left
            marker_dict[1][0][1],  # top-right
            marker_dict[2][0][2],  # bottom-right
            marker_dict[3][0][3],  # bottom-left
        ], dtype=np.float32)
    except KeyError:
        return None  # not all markers found

    width, height = 800, 600
    pts_dst = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(frame, M, (width, height))
    return warped

# --- Extract center region for color detection ---
def get_center_color(screen_frame, roi_size=100):
    h, w = screen_frame.shape[:2]
    cx, cy = w//2, h//2
    roi = screen_frame[cy-roi_size:cy+roi_size, cx-roi_size:cx+roi_size]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # masks
    red_mask = cv2.inRange(hsv, (0,100,100), (10,255,255)) | cv2.inRange(hsv, (160,100,100), (179,255,255))
    white_mask = cv2.inRange(hsv, (0,0,200), (180,30,255))
    black_mask = cv2.inRange(hsv, (0,0,0), (180,255,50))
    green_mask = cv2.inRange(hsv, (40,50,50), (80,255,255))
    blue_mask  = cv2.inRange(hsv, (100,150,0), (140,255,255))

    counts = {
        "red": int(cv2.countNonZero(red_mask)),
        "white": int(cv2.countNonZero(white_mask)),
        "black": int(cv2.countNonZero(black_mask)),
        "green": int(cv2.countNonZero(green_mask)),
        "blue": int(cv2.countNonZero(blue_mask)),
    }
    return max(counts, key=counts.get)

# --- Main receiver ---
def receive_message(source=0, roi_size=100, verbose=True):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Cannot open source.")
        return ""

    frames_to_decode = []
    waiting_for_start = True
    message = ""

    print("Receiver started — waiting for START frame...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        screen = detect_screen(frame)
        if screen is None:
            cv2.putText(frame, "Markers not detected", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Receiver", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        color = get_center_color(screen, roi_size)

        # Visualization
        cv2.imshow("Receiver", screen)
        cv2.putText(frame, f"Detected: {color}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # --- Start / End frame handling ---
        if waiting_for_start:
            if detect_start_frame(screen):
                waiting_for_start = False
                if verbose:
                    print("Start frame detected — beginning decoding!")
            continue

        if detect_end_frame(screen):
            if verbose:
                print("End frame detected — stopping capture.")
            break

        # Collect frames for decoding
        frames_to_decode.append(screen)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # --- Decode message ---
    bits = decode_bits_with_blue(frames_to_decode, roi_size=roi_size, verbose=verbose)
    message = bits_to_message(bits)
    if verbose:
        print("Final message:", message)
    return message

# --- Run ---
if __name__ == "__main__":
    decoded_msg = receive_message(source=0, verbose=True)
    print("Decoded message:", decoded_msg)
