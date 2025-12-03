
# --- Imports ---

import cv2
import numpy as np

# --- Functions ---

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

def roi_alignment(frame, inset_px = 0):

    h, w = frame.shape[:2]
    w_px = 0
    h_px = 0
    roi_coordinates = None
    _, corners, ids = detect_screen(frame)

    if corners is not None and ids is not None and len(ids) > 0:
        ids_flat = ids.flatten() if hasattr(ids, "flatten") else np.array(ids).flatten()
        id_to_corners = {int(marker_id): corners[idx][0] for idx, marker_id in enumerate(ids_flat)}

        required_ids = [0, 1, 2, 3]
        if all(i in id_to_corners for i in required_ids):

            # Size of markers
            pts = id_to_corners[0]
            w_px = np.linalg.norm(pts[1] - pts[0])  # width in pixels
            h_px = np.linalg.norm(pts[2] - pts[1])  # height in pixels

            # Collect all corners from the four markers
            all_corners = np.vstack([id_to_corners[i] for i in required_ids])
            x0, y0 = np.min(all_corners, axis=0) + inset_px
            x1, y1 = np.max(all_corners, axis=0) - inset_px

            # Clip to frame
            x0, x1 = max(0, int(x0)), min(w, int(x1))
            y0, y1 = max(0, int(y0)), min(h, int(y1))

            if x1 - x0 > 5 and y1 - y0 > 5:
                roi_coordinates = (x0, x1, y0, y1)
                print("ROI set around outer corners of markers.")

    return roi_coordinates, w_px, h_px

saved_corners = {0: None, 1: None} 

def roi_alignment_for_large_markers(corners, marker_ids, frame):

    """
    Creates a ROI around the outer corners of the ArUco markers.

    Arguments:
        "corners":
        "marker_ids":
        "frame":

    Returns:
        "roi_coordinates":
        "w_px":
        "h_px":

    """
    
    global saved_corners
    
    h, w = frame.shape[:2]
    w_px = 0
    h_px = 0
    roi_coordinates = None

    if hasattr(marker_ids, "flatten"):
        ids_flat = marker_ids.flatten()
    
    else:
        ids_flat = np.array(marker_ids).flatten()

    id_to_corners = {}
    
    for idx, marker_id in enumerate(ids_flat):
        id_to_corners[int(marker_id)] = corners[idx][0]

    for marker_id in [0, 1]:
        if marker_id in id_to_corners:
            saved_corners[marker_id] = id_to_corners[marker_id]

    if saved_corners[0] is not None and saved_corners[1] is not None:
        
        pts = saved_corners[0]

        w_px = np.linalg.norm(pts[1] - pts[0])
        h_px = np.linalg.norm(pts[2] - pts[1])

        all_corners = np.vstack([saved_corners[0], saved_corners[1]])

        x0, y0 = np.min(all_corners, axis = 0) 
        x1, y1 = np.max(all_corners, axis = 0) 

        x0, x1 = max(0, int(x0)), min(w, int(x1))
        y0, y1 = max(0, int(y0)), min(h, int(y1))

        if x1 - x0 > 5 and y1 - y0 > 5:
            roi_coordinates = (x0, y0, x1, y1)
            print("\n[INFO] ROI set around outer corners of markers.")

    return roi_coordinates, w_px, h_px
