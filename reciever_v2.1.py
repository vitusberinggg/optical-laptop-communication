import cv2
import numpy as np
from utilities.detection_functions import detect_start_frame, detect_end_frame
from utilities.decoding_functions import decode_bits_with_blue, bits_to_message

def detect_screen(frame):
    """Detects and warps the screen using four ArUco markers."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(frame)
    else:
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=params)

    if ids is None or len(ids) < 4:
        return None

    id_to_corners = {int(i[0]): c for i, c in zip(ids, corners)}
    if len(id_to_corners) < 4:
        return None

    # Order markers: top-left, top-right, bottom-right, bottom-left
    centers = [(mid, c[0].mean(axis=0)) for mid, c in id_to_corners.items()]
    centers_sorted = sorted(centers, key=lambda x: (x[1][1], x[1][0]))
    top_two = sorted(centers_sorted[:2], key=lambda x: x[1][0])
    bottom_two = sorted(centers_sorted[2:], key=lambda x: x[1][0])
    ordered = [top_two[0][1], top_two[1][1], bottom_two[1][1], bottom_two[0][1]]
    pts_src = np.array(ordered, dtype=np.float32)

    width, height = 800, 600
    pts_dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(frame, M, (width, height))
    return warped

def receive_message_debug(source=0, roi_size=150, verbose=True):
    """Debug receiver for a video file or webcam showing ROI and warped screen."""
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

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_video_file:
                    print("End of video file.")
                    break
                continue

            # Detect screen
            screen = detect_screen(frame)
            if screen is None:
                if verbose:
                    print("Warning: Could not detect all ArUco markers.")
                cv2.imshow("Raw video", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Warped screen copy for drawing
            display = screen.copy()
            h, w = screen.shape[:2]
            cx, cy = w//2, h//2

            # Safe ROI coordinates
            x0, y0 = max(0, cx - roi_size), max(0, cy - roi_size)
            x1, y1 = min(w, cx + roi_size), min(h, cy + roi_size)
            roi = screen[y0:y1, x0:x1]

            # Draw rectangle on warped screen
            cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 255), 2)

            # Draw rectangle on original raw frame for reference
            orig_h, orig_w = frame.shape[:2]
            cv2.rectangle(frame, (int(orig_w/2 - roi_size), int(orig_h/2 - roi_size)),
                                 (int(orig_w/2 + roi_size), int(orig_h/2 + roi_size)),
                                 (0, 255, 255), 2)

            # Show both
            cv2.imshow("Warped screen (debug)", display)
            cv2.imshow("Raw video with ROI", frame)
            cv2.imshow("ROI being checked", roi)

            # Debug info
            if verbose:
                avg_bgr = roi.mean(axis=(0,1))
                distance_to_green = np.linalg.norm(avg_bgr - np.array([0,255,0]))
                print(f"ROI avg BGR: {avg_bgr.round(1)}, Distance to green: {distance_to_green:.1f}")
                print("detect_start_frame(screen) ->", detect_start_frame(screen))

            # Start frame detection
            if waiting_for_start and detect_start_frame(screen):
                print("Start frame detected — beginning capture!")
                waiting_for_start = False

            # End frame detection
            if detect_end_frame(screen):
                print("End frame detected — stopping capture.")
                break

            frames.append(screen)

            key = cv2.waitKey(delay_ms if is_video_file else 1) & 0xFF
            if key == ord('q'):
                print("User requested exit.")
                break

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received — exiting gracefully.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Decode collected frames
    if verbose:
        print(f"Decoding {len(frames)} frames...")
    bits = decode_bits_with_blue(frames, roi_size=roi_size, verbose=verbose)
    message = bits_to_message(bits)
    if verbose:
        print("Final message:", message)
    return message

if __name__ == "__main__":
    # Example: video file
    receive_message_debug(source=r"C:\Users\eanpaln\Videos\Screen Recordings\Recordinggg.mp4",
                          roi_size=150,
                          verbose=True)
