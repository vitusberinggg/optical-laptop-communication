import cv2
import numpy as np
from utilities.detection_functions import detect_start_frame, detect_end_frame
from utilities.decoding_functions import decode_bits_with_blue, bits_to_message

def detect_screen(frame):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(frame)
    else:
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=params)

    display = frame.copy()
    if corners is not None and ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(display, corners, ids)
    return display, corners, ids

def receive_message_webcam(webcam_index=0, inset_px=0, verbose=True):
    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        print("Error: Could not open webcam", webcam_index)
        return ""

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30
    delay_ms = max(1, int(1000 / fps))

    frames = []
    waiting_for_start = True
    roi_coords = None  # Store ROI after detecting markers

    print("Webcam receiver started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam.")
            break

        display = frame.copy()
        h, w = frame.shape[:2]

        # Detect markers until ROI is set
        if roi_coords is None:
            display, corners, ids = detect_screen(frame)

            if corners is not None and ids is not None and len(ids) > 0:
                # Flatten IDs
                ids_flat = ids.flatten() if hasattr(ids, "flatten") else np.array(ids).flatten()

                # Map ID to corner coordinates
                id_to_corners = {int(m_id): corners[idx][0] for idx, m_id in enumerate(ids_flat)}

                required_ids = [0, 1, 2, 3]
                if all(i in id_to_corners for i in required_ids):
                    # Collect all corner points of the required markers
                    all_corners = np.vstack([id_to_corners[i] for i in required_ids])
                    x_min, y_min = np.min(all_corners, axis=0)
                    x_max, y_max = np.max(all_corners, axis=0)

                    # Apply optional inset
                    x0, x1 = max(0, int(x_min) + inset_px), min(w, int(x_max) - inset_px)
                    y0, y1 = max(0, int(y_min) + inset_px), min(h, int(y_max) - inset_px)

                    if x1 - x0 > 5 and y1 - y0 > 5:
                        roi_coords = (x0, x1, y0, y1)  # Store ROI

        # Fallback ROI in center if markers never detected
        if roi_coords is None:
            cx, cy = w // 2, h // 2
            roi_size = 150
            x0, x1 = max(0, cx - roi_size), min(w, cx + roi_size)
            y0, y1 = max(0, cy - roi_size), min(h, cy + roi_size)
            roi_coords = (x0, x1, y0, y1)
        else:
            x0, x1, y0, y1 = roi_coords

        # Draw ROI rectangle
        cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 255), 2)

        # Extract ROI
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            roi = np.zeros((10, 10, 3), dtype=np.uint8)

        # Average color debug
        avg_color = roi.mean(axis=(0, 1)).round(1)
        cv2.putText(display, f"ROI avg BGR: {avg_color}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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

    bits = decode_bits_with_blue(frames, roi_size=roi_coords[1]-roi_coords[0], verbose=verbose)
    message = bits_to_message(bits)
    if verbose:
        print("Final message:", message)
    return message

if __name__ == "__main__":
    receive_message_webcam(webcam_index=0, inset_px=0, verbose=True)
