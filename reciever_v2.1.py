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

    return corners, ids

def receive_message_fast(source=0, roi_size=150, inset_px=30, verbose=False, show_frames=False):
    """
    Fast receiver for messages, optimized for speed.
    source: webcam index or path to video file
    roi_size: fallback ROI size if markers not detected
    inset_px: inset inside marker bounding box
    verbose: prints bit info
    show_frames: optionally show frames for debugging
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error opening source:", source)
        return ""

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay_ms = max(1, int(1000 / fps))
    is_video_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0

    frames = []
    waiting_for_start = True
    frame_idx = 0

    if verbose:
        print("Fast receiver started.")

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_video_file:
                if verbose:
                    print("End of video file.")
                break
            continue

        frame_idx += 1

        corners, ids = detect_screen(frame)
        h, w = frame.shape[:2]

        # --- Compute ROI based on markers if available ---
        roi_valid = False
        if corners is not None and ids is not None and len(ids) > 0:
            try:
                ids_flat = ids.flatten()
            except:
                ids_flat = np.array(ids).flatten()

            id_to_center = {}
            for idx, m_id in enumerate(ids_flat):
                c = corners[idx][0]  # shape (4,2)
                center = c.mean(axis=0)
                id_to_center[int(m_id)] = center

            required_ids = [0, 1, 2, 3]
            if all(i in id_to_center for i in required_ids):
                centers = np.array([id_to_center[i] for i in required_ids])
                xs = centers[:, 0]
                ys = centers[:, 1]

                bx0 = int(np.min(xs))
                bx1 = int(np.max(xs))
                by0 = int(np.min(ys))
                by1 = int(np.max(ys))

                x0 = max(0, min(w-1, bx0 + inset_px))
                x1 = max(0, min(w, bx1 - inset_px))
                y0 = max(0, min(h-1, by0 + inset_px))
                y1 = max(0, min(h, by1 - inset_px))

                if x1 - x0 > 5 and y1 - y0 > 5:
                    roi_valid = True

        # --- Fallback ROI ---
        if not roi_valid:
            cx, cy = w // 2, h // 2
            x0, x1 = max(0, cx - roi_size), min(w, cx + roi_size)
            y0, y1 = max(0, cy - roi_size), min(h, cy + roi_size)

        # Extract ROI safely
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            roi = np.zeros((10, 10, 3), dtype=np.uint8)

        # Start frame detection
        if waiting_for_start and detect_start_frame(frame):
            if verbose:
                print(f"Frame {frame_idx}: Start frame detected.")
            waiting_for_start = False

        # End frame detection
        if detect_end_frame(frame):
            if verbose:
                print(f"Frame {frame_idx}: End frame detected.")
            break

        frames.append(frame)

        if show_frames:
            display = frame.copy()
            cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 255), 2)
            cv2.imshow("Fast Receiver Debug", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    if show_frames:
        cv2.destroyAllWindows()

    bits = decode_bits_with_blue(frames, roi_size=roi_size, verbose=verbose)
    message = bits_to_message(bits)

    if verbose:
        print("Final message:", message)

    return message
