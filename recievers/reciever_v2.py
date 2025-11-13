# --- Imports ---
import cv2
import numpy as np

# --- Helper: detect dominant color ---
def read_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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
        [0,0],
        [width,0],
        [width,height],
        [0,height]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(frame, M, (width, height))
    return warped

# --- Extract center region for color detection ---
def get_center_color(screen_frame):
    h, w = screen_frame.shape[:2]
    size = 100
    cx, cy = w//2, h//2
    roi = screen_frame[cy-size:cy+size, cx-size:cx+size]
    return read_color(roi)

# --- Main receiver ---
def receive_message():
    cap = cv2.VideoCapture(0)  # webcam
    bits = ""
    message = ""
    last_color = None
    waiting_for_sync = True
    decoding = False
    bit_ready = False

    print("Receiver started — waiting for GREEN to sync...")

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

        color = get_center_color(screen)

        # Visualization
        cv2.imshow("Receiver", screen)
        cv2.putText(frame, f"Detected: {color}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # --- SYNC / DECODING ---
        if waiting_for_sync:
            if color == "green" and last_color != "green":
                print("Green detected — syncing...")
            elif color != "green" and last_color == "green":
                print("Green ended — starting decoding!")
                waiting_for_sync = False
                decoding = True
                bit_ready = True

        elif decoding:
            if color == "blue":
                bit_ready = True
            elif color in ["white","black"] and bit_ready:
                bits += "1" if color == "white" else "0"
                print(f"Bit: {bits[-1]}")
                bit_ready = False
            elif color == "red" and last_color != "red":
                while len(bits) >= 8:
                    byte = bits[:8]
                    bits = bits[8:]
                    try:
                        ch = chr(int(byte,2))
                    except:
                        ch = '?'
                    message += ch
                    print(f"Received char: {ch}")
                if 0 < len(bits) < 8:
                    byte = bits.ljust(8,'0')
                    try:
                        ch = chr(int(byte,2))
                    except:
                        ch = '?'
                    message += ch
                    print(f"Received char (padded): {ch}")
                bits = ""

        last_color = color
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    print("Final message:", message)
    cap.release()
    cv2.destroyAllWindows()

# --- Run ---
if __name__ == "__main__":
    receive_message()

