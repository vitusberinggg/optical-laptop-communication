import cv2
import numpy as np
from utilities.detection_functions import detect_start_frame, detect_end_frame
from utilities.decoding_functions import bits_to_message, decode_bits_with_blue

def detect_screen(frame):
    # (use whichever version you already have that supports both OpenCV versions)
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

    marker_dict = {id[0]: corner for id, corner in zip(ids, corners)}
    try:
        pts_src = np.array([
            marker_dict[0][0][0],
            marker_dict[1][0][1],
            marker_dict[2][0][2],
            marker_dict[3][0][3],
        ], dtype=np.float32)
    except KeyError:
        return None

    width, height = 800, 600
    pts_dst = np.array([[0,0],[width,0],[width,height],[0,height]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(frame, M, (width, height))
    return warped

def receive_message(source=0, roi_size=100, verbose=True):
    """
    source: int for webcam (0...), or string path to video file
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Cannot open source:", source)
        return ""

    # get fps and compute wait delay (ms). if fps is 0 (unknown), use sensible defaults
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0 if isinstance(source, int) else 15.0
    delay_ms = max(1, int(1000 / fps))

    # detect whether this is a video file (has a frame count)
    is_video_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0

    frames_to_decode = []
    waiting_for_start = True

    if verbose:
        print(f"Receiver started — waiting for START frame... (source={source}, fps={fps:.2f}, delay_ms={delay_ms})")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # if reading from a file, EOF -> stop capturing
                if is_video_file:
                    if verbose:
                        print("End of video file reached.")
                    break
                # webcam sometimes returns False briefly; continue attempting
                continue

            # Optional: if the frame is huge, you can resize for performance:
            # frame = cv2.resize(frame, (1280, 720))

            screen = detect_screen(frame)
            if screen is None:
                # show raw frame so user can position the sender
                cv2.imshow("Receiver", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # show aligned screen (warped)
            cv2.imshow("Receiver", screen)

            # START detection
            if waiting_for_start:
                if detect_start_frame(screen):
                    waiting_for_start = False
                    if verbose:
                        print("Start frame detected — beginning capture!")
                # use small waitKey while waiting for start for responsiveness
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # END detection
            if detect_end_frame(screen):
                if verbose:
                    print("End frame detected — stopping capture.")
                break

            # collect frames for decoding
            frames_to_decode.append(screen)

            # choose waitKey delay depending on source type:
            # - video: use file frame delay to keep timing similar to encoded message
            # - webcam: small delay for interactivity
            if is_video_file:
                key = cv2.waitKey(delay_ms) & 0xFF
            else:
                key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                if verbose: print("User requested exit.")
                break

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received — exiting gracefully.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # decode collected frames
    if verbose:
        print(f"Decoding {len(frames_to_decode)} frames...")
    bits = decode_bits_with_blue(frames_to_decode, roi_size=roi_size, verbose=verbose)
    message = bits_to_message(bits)
    if verbose:
        print("Final message:", message)
    return message

# Example usage:
if __name__ == "__main__":
    # webcam:
    # decoded = receive_message(source=0, verbose=True)

    # or video file:
    decoded = receive_message(source=r"C:\Users\eanpaln\Videos\Screen Recordings\Recordingg.mp4", verbose=True)
    print("Decoded:", decoded)
