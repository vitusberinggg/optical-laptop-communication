
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
    Tries to detect the ArUco marker frame shown on the sender screen and map points from one plane to another.

    Arguments:
        "frame": Frame to check for markers.

    Returns:
        "homography_matrix": A 3 x 3 transformation matrix.

    """

    aruco_marker_detector_parameters = cv2.aruco.DetectorParameters()
    aruco_marker_detector = cv2.aruco.ArucoDetector(aruco_marker_dictionary, aruco_marker_detector_parameters)

    frame_corners, aruco_marker_ids, _ = aruco_marker_detector.detectMarkers(frame)

    if aruco_marker_ids is None or len(aruco_marker_ids) < 4:
        return None
    
    aruco_marker_ids = aruco_marker_ids.flatten()

#   Sorting the markers by ID

    sorted_marker_ids = [None] * 4

    for frame_corner, marker_id in zip(frame_corners, aruco_marker_ids):

        if marker_id < 4:
            sorted_marker_ids[marker_id] = frame_corner[0]
    
    if any(corner is None for corner in sorted_marker_ids):
        return None
    
#   Perspective mapping
    
    source = np.array([ # Marker coordinates from the camera frame
        sorted_marker_ids[0],
        sorted_marker_ids[1],
        sorted_marker_ids[3],
        sorted_marker_ids[2]],
        dtype = np.float32)
    
    destination_layout = np.array([ # What the markers should form
        [0, 0],
        [sender_output_width, 0],
        [sender_output_width, sender_output_height],
        [0, sender_output_height]],
        dtype = np.float32)
    
    homography_matrix = cv2.getPerspectiveTransform(source, destination_layout)

    return homography_matrix

def detect_reference_image(potential_sync_frame, reference_image):

    """
    Detects if the reference image is present in the potential sync frame.

    Arguments:
        "potential_sync_frame" (np.ndarray): A NumPy array representing the potential sync frame pixels.
        "reference_image" (np.ndarray): A NumPy array representing the reference image pixels.

    Returns:
        tuple: A tuple containing a boolean indicating if the reference image is detected and the maximum correlation value.

    """

    resized_frame = cv2.resize(potential_sync_frame, (reference_image.shape[1], reference_image.shape[0])) # Resize the potential sync frame to match the reference image size

    correlation = cv2.matchTemplate(resized_frame, reference_image, cv2.TM_CCOEFF_NORMED) # Perform template matching to find the reference image in the potential sync frame

    max_correlation = cv2.minMaxLoc(correlation)[1] # Get the maximum correlation value

    return max_correlation > reference_match_threshold, max_correlation # Return whether the reference image is detected and the maximum correlation value

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