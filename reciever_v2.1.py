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
    if corners is not None and ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(display, corners, ids)
    else:
        cv2.putText(display, "No markers detected", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    return display, corners, ids


def receive_message_debug(source=0, roi_size=150, inset_px=30, verbose=True):
    """
    source: webcam index or path to video file
    roi_size: used only for fallback center ROI
    inset_px: how many pixels to inset the ROI inside the marker bounding box
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

    print("Debug receiver started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_video_file:
                print("End of video file.")
                break
            continue

        display, corners, ids = detect_screen(frame)

        h, w = frame.shape[:2]

        # --- Compute ROI based on markers if available ---
        roi_valid = False
        if corners is not None and ids is not None and len(ids) > 0:
            try:
                ids_flat = ids.flatten()
            except:
                ids_flat = np.array(ids).flatten()

            # compute marker centers and map by id
            id_to_center = {}
            for idx, m_id in enumerate(ids_flat):
                c = corners[idx][0]  # shape (4,2)
                center = c.mean(axis=0)  # (x, y)
                id_to_center[int(m_id)] = center

            # we expect four corner markers with ids 0..3 (adjust if your ids differ)
            required_ids = [0,1,2,3]
            if all(i in id_to_center for i in required_ids):
                centers = np.array([id_to_center[i] for i in required_ids])  # order 0,1,2,3
                xs = centers[:,0]
                ys = centers[:,1]

                # bounding box that contains the four marker centers
                bx0 = int(np.min(xs))
                bx1 = int(np.max(xs))
                by0 = int(np.min(ys))
                by1 = int(np.max(ys))

                # inset the ROI so it sits inside the markers (avoid overlap)
                x0 = bx0 + inset_px
                x1 = bx1 - inset_px
                y0 = by0 + inset_px
                y1 = by1 - inset_px

                # Clip to image bounds and ensure valid
                x0 = max(0, min(w-1, x0))
                x1 = max(0, min(w, x1))
                y0 = max(0, min(h-1, y0))
                y1 = max(0, min(h, y1))

                if x1 - x0 > 5 and y1 - y0 > 5:
                    roi_valid = True
                else:
                    # computed ROI too small -> fallback
                    roi_valid = False

        # --- Fallback: center ROI as before ---
        if not roi_valid:
            cx, cy = w//2, h//2
            x0, x1 = cx - roi_size, cx + roi_size
            y0, y1 = cy - roi_size, cy + roi_size
            # clip
            x0 = max(0, x0); x1 = min(w, x1)
            y0 = max(0, y0); y1 = min(h, y1)

        # Draw ROI rectangle on the display image
        cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 255), 2)

        # Extract ROI safely
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            # create a small black placeholder if ROI invalid
            roi = np.zeros((10,10,3), dtype=np.uint8)

        # Compute average color in ROI and overlay debug text
        avg_color = roi.mean(axis=(0,1)).round(1)
        cv2.putText(display, f"ROI avg BGR: {avg_color}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Status text
        status = "Waiting for START" if waiting_for_start else "Decoding..."
        cv2.putText(display, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Debug Receiver", display)
        cv2.imshow("ROI", roi)

        # Start frame detection (note: you used frame previously; keep same)
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
    receive_message_debug(source=r"C:\Users\eanpaln\Videos\Screen Recordings\Screen Recording 2025-11-14 153114.mp4",
                          roi_size=150, inset_px=40, verbose=True)
