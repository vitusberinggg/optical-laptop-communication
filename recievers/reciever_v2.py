# --- Imports ---
import cv2
from utilities.detect_end_frame import detect_end_frame
from utilities.bits_to_message import bits_to_message
from utilities.detect_start_frame import detect_start_frame
from utilities.decode_bits_with_blue import decode_bits_with_blue

# --- Detect sender screen using ArUco markers ---
def detect_screen(frame):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    if ids is None or len(ids) < 4:
        return None  # not enough markers detected

    marker_dict = {id[0]: corner for id, corner in zip(ids, corners)}

    try:
        pts_src = [
            marker_dict[0][0][0],  # top-left
            marker_dict[1][0][1],  # top-right
            marker_dict[2][0][2],  # bottom-right
            marker_dict[3][0][3],  # bottom-left
        ]
    except KeyError:
        return None  # not all markers found

    pts_src = cv2.convertPointsToHomogeneous(np.array(pts_src, dtype='float32'))[:, 0, :2]
    width, height = 800, 600
    pts_dst = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype='float32')

    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(frame, M, (width, height))
    return warped

# --- Main receiver ---
def receive_message(source=0, roi_size=100, verbose=True):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Cannot open source.")
        return ""

    frames_to_decode = []
    waiting_for_start = True

    if verbose:
        print("Receiver started — waiting for START frame...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        screen = detect_screen(frame)
        if screen is None:
            cv2.imshow("Receiver", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        cv2.imshow("Receiver", screen)

        # --- Start frame detection ---
        if waiting_for_start:
            if detect_start_frame(screen):
                waiting_for_start = False
                if verbose:
                    print("Start frame detected — beginning capture!")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # --- End frame detection ---
        if detect_end_frame(screen):
            if verbose:
                print("End frame detected — stopping capture.")
            break

        # --- Collect frames for decoding ---
        frames_to_decode.append(screen)

        if cv2.waitKey(1) & 0xFF == ord('q'):
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
