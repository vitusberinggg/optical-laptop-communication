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

    if ids is None or len(ids) < 4:
        return None, corners, ids

    # Draw detected markers for debugging
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    id_to_corners = {int(i[0]): c for i, c in zip(ids, corners)}
    centers = [(mid, c[0].mean(axis=0)) for mid, c in id_to_corners.items()]
    if len(centers) < 4:
        return None, corners, ids

    # Order points for perspective transform
    centers_sorted = sorted(centers, key=lambda x: (x[1][1], x[1][0]))
    top_two = sorted(centers_sorted[:2], key=lambda x: x[1][0])
    bottom_two = sorted(centers_sorted[2:4], key=lambda x: x[1][0])
    ordered = [top_two[0][1], top_two[1][1], bottom_two[1][1], bottom_two[0][1]]
    pts_src = np.array(ordered, dtype=np.float32)

    width, height = 800, 600
    pts_dst = np.array([[0,0],[width,0],[width,height],[0,height]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(frame, M, (width, height))
    return warped, corners, ids

def receive_message_debug(source=0, roi_size=150, verbose=True):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error opening source:", source)
        return ""

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0
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

        screen, corners, ids = detect_screen(frame)

        # Show original frame with ArUco markers
        display_raw = frame.copy()
        if corners is not None and ids is not None:
            cv2.aruco.drawDetectedMarkers(display_raw, corners, ids)
        cv2.imshow("Receiver (raw + markers)", display_raw)

        if screen is None:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        display_warped = screen.copy()
        h, w = screen.shape[:2]
        cx, cy = w//2, h//2
        x0, x1 = cx - roi_size, cx + roi_size
        y0, y1 = cy - roi_size, cy + roi_size

        # Draw square on warped screen for ROI
        cv2.rectangle(display_warped, (x0, y0), (x1, y1), (0, 255, 255), 2)

        roi = screen[y0:y1, x0:x1]
        avg = roi.mean(axis=(0,1))  # BGR
        if verbose:
            print(f"ROI avg color (B,G,R): {avg.round(1)}")
            print("detect_start_frame(screen) ->", detect_start_frame(screen))

        cv2.imshow("Warped screen (ROI debug)", display_warped)
        cv2.imshow("ROI being checked", roi)

        if waiting_for_start and detect_start_frame(screen):
            print("Start frame detected — beginning capture!")
            waiting_for_start = False

        if detect_end_frame(screen):
            print("End frame detected — stopping capture.")
            break

        frames.append(screen)

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
    receive_message_debug(
        source=r"C:\Users\eanpaln\Videos\Screen Recordings\Recordinggg.mp4",
        roi_size=150,
        verbose=True
    )
