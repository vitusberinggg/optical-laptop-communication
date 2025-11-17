
# --- Imports ---

import cv2
import numpy as np

from utilities.global_definitions import (
    sender_output_height, sender_output_width,
    reference_match_threshold,
    start_frame_color, start_frame_detection_tolerance,
    end_frame_color, end_frame_detection_tolerance,
    aruco_marker_dictionary
)

# --- Functions ---

def detect_aruco_marker_frame(frame):
    """
    Detects ArUco markers in the frame and returns a homography matrix
    from detected sender corners to a normalized layout.
    Ensures IDs are in sender's corner order: 0=TL, 1=TR, 2=BR, 3=BL.
    """

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Setup ArUco detector (matches sender dictionary)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_marker_dictionary, aruco_params)

    # Detect markers
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) < 4:
        # Not enough markers detected
        return None

    # Map marker IDs to their corners
    id_to_corner = {id[0]: corner[0] for id, corner in zip(ids, corners)}

    try:
        # Order corners according to sender: 0=TL, 1=TR, 2=BR, 3=BL
        source = np.array([
            id_to_corner[0][0],  # top-left corner of marker 0
            id_to_corner[1][1],  # top-right corner of marker 1
            id_to_corner[2][2],  # bottom-right corner of marker 2
            id_to_corner[3][3],  # bottom-left corner of marker 3
        ], dtype=np.float32)
    except KeyError:
        # Some marker missing, can't compute homography
        return None

    # Destination layout in the receiver frame
    destination_layout = np.array([
        [0, 0],
        [sender_output_width, 0],
        [sender_output_width, sender_output_height],
        [0, sender_output_height]
    ], dtype=np.float32)

    # Compute homography
    homography_matrix = cv2.getPerspectiveTransform(source, destination_layout)
    return homography_matrix


def detect_ecc_reference_image(potential_sync_frame, ecc_reference_image):

    """
    Detects if the reference image is present in the potential sync frame.

    Arguments:
        "potential_sync_frame" (np.ndarray): A NumPy array representing the potential sync frame pixels.
        "ecc_reference_image" (np.ndarray): A NumPy array representing the reference image pixels.

    Returns:
        tuple: A tuple containing a boolean indicating if the reference image is detected and the maximum correlation value.

    """

    resized_frame = cv2.resize(potential_sync_frame, (ecc_reference_image.shape[1], ecc_reference_image.shape[0])) # Resize the potential sync frame to match the reference image size

    grayscale_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    correlation = cv2.matchTemplate(grayscale_resized_frame, ecc_reference_image, cv2.TM_CCOEFF_NORMED) # Perform template matching to find the reference image in the potential sync frame

    max_correlation = cv2.minMaxLoc(correlation)[1] # Get the maximum correlation value

    return max_correlation > reference_match_threshold # Return True if the correlation is larger than the threshold, False if not

def detect_start_frame(frame):

    """
    Detects if the average frame color is close to the start frame color.

    Arguments:
        "frame" (np.ndarray): A NumPy array representing the frame pixels.

    Returns:
        bool: True if the average frame color is close to the start frame color, False otherwise.
    
    """

    average_frame_color = frame.mean(axis = (0, 1)) # Calculates the average color of the frame

    distance = np.linalg.norm(average_frame_color - np.array(start_frame_color)) # Calculates the Euclidean distance between the average frame color and the start frame color

    return distance < start_frame_detection_tolerance # Returns True if the distance is less than the tolerance, False otherwise

def detect_end_frame(frame):

    """
    Detects if the average frame color is close to the end frame color.

    Arguments:
        "frame" (np.ndarray): A NumPy array representing the frame pixels.

    Returns:
        bool: True if the average frame color is close to the end frame color, False otherwise.
    
    """

    average_frame_color = frame.mean(axis = (0, 1)) # Calculates the average color of the frame

    distance = np.linalg.norm(average_frame_color - np.array(end_frame_color)) # Calculates the Euclidean distance between the average frame color and the end frame color

    return distance < end_frame_detection_tolerance # Returns True if the distance is less than the tolerance, False otherwise