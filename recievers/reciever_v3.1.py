
# --- Imports ---

import cv2
import time
import numpy as np

from utilities.generate_reference_image import generate_reference_image
from utilities.detect_end_frame import detect_end_frame
from utilities.detect_reference_image import detect_reference_image
from utilities.decode_bitgrid import decode_bitgrid
from utilities.bits_to_message import bits_to_message
from utilities.global_definitions import reference_image_duration, frame_duration

# --- Definitions ---

camera_index = 0 # Index of the camera to be used

# --- Main function ---

def receive_message():

    """
    Receives a message from the sender.
    
    Arguments:
        None
        
    Returns:
        None
    
    """

    videoCapture = cv2.VideoCapture(camera_index) # Initialize video capture from the specified camera index

    reference_image = generate_reference_image() # Generate the reference image
    reference_image_height, reference_image_width = reference_image.shape # Get the dimensions of the reference image
    reference_image_float32 = reference_image.astype(np.float32) # Converts the reference image to float32 format

    ecc_warp_mode = cv2.MOTION_AFFINE # Sets the ECC warp mode to affine
    ecc_warp_matrix = np.eye(2, 3, dtype = np.float32) # Initializes the ECC warp matrix as an identity matrix
    ecc_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-7) # Sets the ECC criteria for the alignment process

    reference_image_detected = False
    reference_start_time = None
    ecc_computed = False

    bitstring = ""
    frame_counter = 0
    end_frame_detected = False

    print("[INFO] Waiting for the reference frame...")

    while True:

        snapshots = []

        for _ in range(samples_per_frame): # For each sample:

            for _ in range(2):
                videoCapture.grab() # Grab two frames to ensure the latest frame is captured (mini buffer flush)

            return_value, frame = videoCapture.read() # Capture a frame from the video feed

            if not return_value: # If the frame capture was unsuccessful:
                continue # Skip to the next iteration

            frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 

            snapshots.append(frame_grayscale.astype(np.float32))

            time.sleep(sample_space)

        if not snapshots: # If there aren't any snapshots:
            continue # Skip to the next iteration

        average_grayscaled_frame = np.mean(snapshots, axis=0).astype(np.float32)

        if not reference_image_detected: # If no reference image has been detected:
            if (average_grayscaled_frame.shape[1], average_grayscaled_frame.shape[0]) != (reference_image_width, reference_image_height): # If the captured frame isn't the same size as the reference image:

    videoCapture.release()
    cv2.destroyAllWindows()

    if end_frame_detected: # If the end frame is detected:
        message = bits_to_message(bitstring) # Decode the bitstrig
        print(message) # Print the decoded message

    else:
        print("\n[INFO] No end frame detected, transmission may be incomplete.")

# --- Execution ---

if __name__ == "__main__": # If the script is run directly:
    receive_message() # Call the main function to receive the message