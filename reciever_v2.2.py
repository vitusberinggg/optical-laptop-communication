import cv2
import numpy as np
from utilities.detection_functions import detect_start_frame, detect_end_frame
from utilities.decoding_functions import decode_bits_with_blue, bits_to_message

def detect_screen(frame):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # Use new ArUco API if available
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(frame)
    else:
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=params)

    # Draw detected markers
    display = frame.copy()
    if corners is not None and ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(display, corners, ids)
    else:
        cv2.putText(display, "No markers detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return display, corners, ids


def receive_message_webcam(webcam_index=0, roi_size=150, inset_px=30, verbose=True):
    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        print("Error: Could not open webcam", webcam_index)
        return ""

    # Use default webcam fps or 30 if unavailable
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30
    delay_ms = max(1, int(1000 / fps))

    frames = []
    waiting_for_start = True

    print("Webcam receiver started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam.")
            break

        display, corners, ids = detect_screen(frame)
        h, w = frame.shape[:2]

        # --- Compute ROI based on markers if available ---
        roi_valid = False
        if corners is not None and ids is not None and len(ids) > 0:
            ids_flat = ids.flatten() if hasattr(ids, "flatten") else np.array(ids).flatten()
            id_to_center = {}
            for idx, m_id in enumerate(ids_flat):
                c = corners[idx][0]
                center = c.mean(axis=0)
                id_to_center[int(m_id)] = center

            required_ids = [0, 1, 2, 3]
            if all(i in id_to_center for i in required_ids):
                centers = np.array([id_to_center[i] for i in required_ids])
                xs = centers[:, 0]
                ys = centers[:, 1]
                bx0, bx1 = int(np.min(xs)), int(np.max(xs))
                by0, by1 = int(np.min(ys)), int(np.max(ys))

                x0, x1 = max(0, bx0 + inset_px), min(w, bx1 - inset_px)
                y0, y1 = max(0, by0 + inset_px), min(h, by1 - inset_px)

                if x1 - x0 > 5 and y1 - y0 > 5:
                    roi_valid = True

        # --- Fallback ROI at center ---
        if not roi_valid:
            cx, cy = w // 2, h // 2
            x0, x1 = max(0, cx - roi_size), min(w, cx + roi_size)
            y0, y1 = max(0, cy - roi_size), min(h, cy + roi_size)

        # Draw ROI rectangle
        cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 255), 2)

        # Extract ROI
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            roi = np.zeros((10, 10, 3), dtype=np.uint8)

        # Compute average color for debug
        avg_color = roi.mean(axis=(0, 1)).round(1)
        cv2.putText(display, f"ROI avg BGR: {avg_color}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Status
        status = "Waiting for START" if waiting_for_start else "Decoding..."
        cv2.putText(display, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show frames
        cv2.imshow("Webcam Receiver", display)
        cv2.imshow("ROI", roi)

        # Start/end frame detection
        if waiting_for_start and detect_start_frame(frame):
            print("Start frame detected — beginning capture!")
            waiting_for_start = False

        if detect_end_frame(frame):
            print("End frame detected — stopping capture.")
            break

        frames.append(frame)

        key = cv2.waitKey(delay_ms) & 0xFF
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
    receive_message_webcam(webcam_index=0, roi_size=150, inset_px=40, verbose=True)
