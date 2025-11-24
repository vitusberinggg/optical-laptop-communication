
# --- Imports ---

import cv2
import numpy as np

from utilities.global_definitions import (
    laptop_webcam_pixel_height, laptop_webcam_pixel_width,
    sender_output_height, sender_output_width,
    ecc_allignment_criteria
)

# --- Functions ---

def roi_alignment(frame, inset_px = 0):

    h, w = frame.shape[:2]
    w_px = 0
    h_px = 0
    roi_coords = None
    display, corners, ids = detect_screen(frame)
    if corners is not None and ids is not None and len(ids) > 0:
        ids_flat = ids.flatten() if hasattr(ids, "flatten") else np.array(ids).flatten()
        id_to_corners = {int(m_id): corners[idx][0] for idx, m_id in enumerate(ids_flat)}

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
                roi_coords = (x0, x1, y0, y1)
                print("ROI set around outer corners of markers.")
    return roi_coords, w_px, h_px

def roi_alignment2(frame, inset_px = 0):
    return # other functions wont work unless this function holds something

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

def create_mask(homography_matrix):

    """
    Creates a binary mask using a homography matrix.

    Arguments:
        "homography_matrix"

    Returns:
        "mask"

    """

    sender_mask = np.full((sender_output_height, sender_output_width), 255, np.uint8)

    inverse_homography_matrix = np.linalg.inv(homography_matrix)

    webcam_mask = cv2.warpPerspective(
        sender_mask,
        inverse_homography_matrix,
        (laptop_webcam_pixel_width, laptop_webcam_pixel_height),
        flags = cv2.INTER_NEAREST
    )

    return webcam_mask

def compute_ecc_transform(reference_image, captured_image):
    
    """
    Alligns the captured reference image to the reference image using ECC (Enhanced Correlation Coefficient).

    Arguments:
        "reference_image": The reference image.
        "captured_image": The image to allign.

    Returns:
        "ecc_warp_matrix": The warp matrix.

    """

    captured_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)

    ecc_warp_matrix = np.eye(2, 3, dtype = np.float32) # Initial warp matrix guess

    try:
        cc, ecc_warp_matrix = cv2.findTransformECC(
            reference_image,
            captured_image,
            ecc_warp_matrix,
            cv2.MOTION_AFFINE,
            ecc_allignment_criteria
        )        

    except cv2.error:

        print("[WARNING] ECC allignment failed.")
        return None
    
    return ecc_warp_matrix
