
# --- Imports ---

import cv2
import time
import numpy as np

from utilities.image_generation_functions import generate_reference_image
from utilities.screen_alignment_functions import create_mask, compute_ecc_transform
from utilities.detection_functions import detect_end_frame, detect_ecc_reference_image, detect_aruco_marker_frame
from utilities.decoding_functions import decode_bitgrid, bits_to_message
from utilities.global_definitions import (
    mask_frame_hsv_lower_limit, mask_frame_hsv_upper_limit,
    total_pixel_count, sender_screen_size_threshold,
    sender_output_height, sender_output_width
)

# --- Definitions ---

camera_index = 0 # Index of the camera to be used
camera = cv2.VideoCapture(camera_index) # Initialize video capture from the specified camera index

# --- Main function ---

def receive_message():

    """
    Receives a message from the sender.
    
    Arguments:
        None
        
    Returns:
        None
    
    """

    print("[INFO] Waiting for ArUco marker frame...")

    while True:

        read_was_successful, frame = camera.read() # Read the frame and the boolean indicating if the read was successful or not
        
        if not read_was_successful: # If the read wasn't successful:
            print("[WARNING] Can't read the camera.")
            continue # Skip this iteration

#       Green screen detection

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Converts the frame to HSV (Hue, Saturation, Value)

        lower_green = np.array(mask_frame_hsv_lower_limit)
        upper_green = np.array(mask_frame_hsv_upper_limit)

        mask_green = cv2.inRange(hsv_frame, lower_green, upper_green) # Creates a mask by setting pixels within the green range to 255, and pixels outside to 0

        green_pixel_count = np.count_nonzero(mask_green) # Counts the amount of green pixels

        if green_pixel_count > (total_pixel_count * sender_screen_size_threshold):

            print(f"[INFO] Green screen covering {green_pixel_count // total_pixel_count} of the frame found.")

            print("[INFO] Creating mask based on lime-green frame area.")

    #       ArUco marker detection

            homography_matrix = detect_aruco_marker_frame(mask_green)

            if homography_matrix is not None:

                print("[INFO] ArUco markers detected in lime-green frame area. Perspective transform locked.")

                aruco_marker_mask = create_mask(homography_matrix)

                print("[INFO] Mask created based on ArUco markers.")

                break

            elif homography_matrix is None:

                print("[WARNING] No ArUco markers detected.")

                continue

        else:

            print(f"[WARNING] No green screen bigger than the size threshold found.")

            continue
    
#   ECC reference image generation

    ecc_reference_image = generate_reference_image() # Generates the reference image

    print("[INFO] Reference image generated. Waiting for reference frame...")

    snapshots = []

    while True:

        read_was_successful, frame = camera.read() # Read the frame and the boolean indicating if the read was successful or not
        
        if not read_was_successful: # If the read wasn't successful:
            continue # Skip this iteration

        frame = cv2.bitwise_and(frame, frame, mask = aruco_marker_mask) # Applies the ArUco marker mask to the captured frame

        reference_images_match = detect_ecc_reference_image(frame, ecc_reference_image) # Checks if the frame matches good enough with the reference image

        if reference_images_match:

            print("[INFO] Sender screen matches reference image, trying to calculate a warp matrix... ")

            ecc_warp_matrix = compute_ecc_transform(ecc_reference_image, frame)

            if ecc_warp_matrix:
                print("[INFO] Warp matrix calculated sucessfully.")
                break

            else:
                print("[WARNING] Failed to calculate warp matrix.")
                continue

        else:
            print("[WARNING] Sender screen doesn't match reference image.")
            continue

    print("[INFO] Ready to recieve data.")

# --- Execution ---

if __name__ == "__main__": # If the script is run directly:
    receive_message() # Call the main function to receive the message