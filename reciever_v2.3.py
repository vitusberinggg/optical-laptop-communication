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


def receive_message_webcam(webcam_index=0, inset_px=30, verbose=True):
    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        print("Error: Could not open webcam", webcam_index)
        return ""

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30
    delay_ms = max(1, int(1000 / fps))

    frames = []
    waiting_for_start = True
    roi_coords = None  # will store (x0, x1, y0, y1) once all markers detected

    print("Webcam receiver started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to ca
