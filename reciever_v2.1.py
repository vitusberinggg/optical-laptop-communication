import cv2
import numpy as np
from utilities.detection_functions import detect_start_frame, detect_end_frame
from utilities.decoding_functions import decode_bits_with_blue, bits_to_message

def detect_screen(frame):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # Check if new ArUco API exists
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(frame)
    else:
        # fallback for older OpenCV
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=params)

    # Draw detected markers if any
    display = frame.copy()
    if corners is not None and ids is not None:
        cv2.aruco.drawDetectedMarkers(display, corners, ids)
    else:
        cv2.putText(display, "No markers detected", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    return display, corners, ids


def receive_message_debug(source=0, roi_size=150, verbose=True):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error opening source:", source)
        return ""

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay_ms = max(1, int(1000 / fps))
    is_video_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0

    frames = []
    waiting_for_start = True

    print("Debug receiver started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_video_file:
                print("End of video file.")
                break
            continue

        display, corners, ids = detect_screen(frame)

        # Draw ROI square at center
        h, w = frame.shape[:2]
        cx, cy = w//2, h//2
        x0, x1 = cx - roi_size, cx + roi_size
        y0, y1 = cy - roi_size, cy + roi_size
        cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 255), 2)

        # Compute average color in ROI
        roi = frame[y0:y1, x0:x1]
        avg_color = roi.mean(axis=(0,1)).round(1)
        cv2.putText(display, f"ROI avg BGR: {avg_color}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Status text
        status = "Waiting for START" if waiting_for_start else "Decoding..."
        cv2.putText(display, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Debug Receiver", display)
        cv2.imshow("ROI", roi)

        # Start frame detection
        if waiting_for_start and detect_start_frame(frame):
            print("Start frame detected — beginning capture!")
            waiting_for_start = False

        # End frame detection
        if detect_end_frame(frame):
            print("End frame detected — stopping capture.")
            break

        frames.append(frame)

        key = cv2.waitKey(delay_ms if is_video_file else 1) & 0xFF
        if key == ord('q'):
            print("User requested exit.")
            break

    cap.release()
    cv2.destroyAllWindows()

    bits = decode_bits_with_blue(frames, roi_size=roi_size, verbose=verbose)
    message = bits_to_message(bits)
    if verbose:
        print("Final message:", message)
    return message

if __name__ == "__main__":
    # Change source to your video file
    receive_message_debug(source=r"C:\Users\eanpaln\Videos\Screen Recordings\Screen Recording 2025-11-14 153114.mp4", roi_size=150, verbose=True)
