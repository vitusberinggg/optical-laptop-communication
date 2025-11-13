
# --- Imports ---

import cv2 # Imports the OpenCV library for image processing
import numpy as np # Imports the NumPy library for numerical operations
import time

from utilities.generate_reference_image import generate_reference_image
from utilities.message_to_frame_bit_arrays import message_to_frame_bit_arrays
from utilities.render_frame import render_frame

from utilities.global_definitions import (
    sender_output_height, sender_output_width,
    reference_image_duration,
    frame_duration)

# ---- Definitions ----

message = "HELLO, THIS IS A TEST MESSAGE!"

# --- Helper functions ---

def create_color_frame(color):

    """
    Creates a solid color frame.

    Arguments:
        "color" (tuple): A tuple representing the BGR color.

    Returns:
        "frame" (np.ndarray): A NumPy array representing the solid color frame pixels.

    """

    return np.full((sender_output_height, sender_output_width, 3), color, dtype = np.uint8)

# --- Main function ---

def send_message(message):

    """
    Sends a message by displaying frames on the screen.

    Arguments:
        "message" (str): The message to be sent.

    Returns:
        None
    
    """

    reference_image = generate_reference_image() # Generates the reference image
    reference_image_bgr = cv2.cvtColor(reference_image, cv2.COLOR_GRAY2BGR) # Converts the reference image to BGR format

    frame_bit_arrays = message_to_frame_bit_arrays(message) # Converts the message to frame bit arrays

    data_frames = []

    for frame_bit_array in frame_bit_arrays: # For each frame bit array:
        rendered_frame = render_frame(frame_bit_array) # Render the frame
        data_frames.append(rendered_frame) # Add the rendered frame to the list of data frames

    end_frame  = create_color_frame((0, 0, 255))

    window = "SENDER" # The name of the OpenCV window
    cv2.namedWindow(window, cv2.WINDOW_NORMAL) # Creates a window with the specified name
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # Sets the window to fullscreen

    start_time = time.time() # Records the start time for the reference image display

    while time.time() - start_time < reference_image_duration: # While the reference image duration limit hasn't been reached:

        cv2.imshow(window, reference_image_bgr) # Display the reference image

        if cv2.waitKey(1) & 0xFF == ord("q"): # If "Q" is pressed:
            cv2.destroyAllWindows() # Close all OpenCV windows
            return # Exit the function
        
        time.sleep(0.001) # Small sleep to prevent high CPU usage

    try:

        for frame in data_frames: # For each frame:

            frame_start_time = time.time() # Records the start time for the current frame

            while time.time() - frame_start_time < frame_duration: # While the frame duration limit hasn't been reached:

                cv2.imshow(window, frame) # Display the current frame in the window

                if cv2.waitKey(1) & 0xFF == ord("q"): # If "Q" is pressed:
                    return # Exit the function
                
                time.sleep(0.001) # Small sleep to prevent high CPU usage

        end_frame_start_time = time.time() # Records the start time for the end frame

        while time.time() - end_frame_start_time < frame_duration: # While the end frame duration limit hasn't been reached:

            cv2.imshow(window, end_frame) # Display the end frame in the window

            if cv2.waitKey(1) & 0xFF == ord("q"): # If "Q" is pressed:
                return # Exit the function
            
            time.sleep(0.001) # Small sleep to prevent high CPU usage
            
    except KeyboardInterrupt: # If a keyboard interrupt occurs (e.g., Ctrl+C):
        pass # Continue to the cleanup section

    finally:
        cv2.destroyAllWindows() # Close all OpenCV windows

if __name__ == "__main__":
    send_message(message)
