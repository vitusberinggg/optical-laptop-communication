
# --- Imports ---

import cv2 # Imports the OpenCV library for image processing
import time

from utilities.image_generation_functions import generate_reference_image, render_frame, create_color_frame, create_aruco_marker_frame
from utilities.encoding_functions import message_to_frame_bit_arrays

from utilities.global_definitions import (
    aruco_marker_frame_duration,
    reference_image_duration,
    frame_duration,
    sync_frame_color, 
    end_frame_color,
    preamble_colors
)

# ---- Definitions ----

message = "HELLO, THIS IS A TEST MESSAGE!"

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

    aruco_marker_frame = create_aruco_marker_frame() # Creates the ArUco marker frame

    preamble_frames = []

    for color in preamble_colors:
        color_frame = create_color_frame(color)
        preamble_frames.append(color_frame)

    frame_bit_arrays = message_to_frame_bit_arrays(message) # Converts the message to frame bit arrays

    data_frames = []

    for frame_bit_array in frame_bit_arrays: # For each frame bit array:
        rendered_frame = render_frame(frame_bit_array) # Render the frame
        data_frames.append(rendered_frame) # Add the rendered frame to the list of data frames

    sync_frame = create_color_frame(sync_frame_color)
    end_frame  = create_color_frame(end_frame_color) # Creates the end frame with the specified color

    window = "SENDER" # The name of the OpenCV window
    cv2.namedWindow(window, cv2.WINDOW_NORMAL) # Creates a window with the specified name
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # Sets the window to fullscreen

#   Aruco marker frame

    aruco_marker_frame_start_time = time.time()

    while time.time() - aruco_marker_frame_start_time < aruco_marker_frame_duration:
        cv2.imshow(window, aruco_marker_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"): # If "Q" is pressed:
            cv2.destroyAllWindows() # Close all OpenCV windows
            return # Exit the function
        
        time.sleep(0.001) # Small sleep to prevent high CPU usage

    for preamble_frame in preamble_frames:
        
        frame_start_time = time.time() # Records the start time for the current frame

        while time.time() - frame_start_time < frame_duration: # While the frame duration limit hasn't been reached:

            cv2.imshow(window, preamble_frame) # Display the current frame in the window

            if cv2.waitKey(1) & 0xFF == ord("q"): # If "Q" is pressed:
                    return # Exit the function
                
            time.sleep(0.001) # Small sleep to prevent high CPU usage

    """

#   ECC reference frame

    reference_image_start_time = time.time() # Records the start time for the reference image display

    while time.time() - reference_image_start_time < reference_image_duration: # While the reference image duration limit hasn't been reached:

        cv2.imshow(window, reference_image_bgr) # Display the reference image

        if cv2.waitKey(1) & 0xFF == ord("q"): # If "Q" is pressed:
            cv2.destroyAllWindows() # Close all OpenCV windows
            return # Exit the function
        
        time.sleep(0.001) # Small sleep to prevent high CPU usage

    """

    try:

#       Data transfer loop

        for frame in data_frames: # For each frame:

            frame_start_time = time.time() # Records the start time for the current frame

            while time.time() - frame_start_time < frame_duration: # While the frame duration limit hasn't been reached:

                cv2.imshow(window, frame) # Display the current frame in the window

                if cv2.waitKey(1) & 0xFF == ord("q"): # If "Q" is pressed:
                    return # Exit the function
                
                time.sleep(0.001) # Small sleep to prevent high CPU usage

            sync_frame_start_time = time.time()

            while time.time() - sync_frame_start_time < frame_duration:

                cv2.imshow(window, sync_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"): # If "Q" is pressed:
                    return # Exit the function
                
                time.sleep(0.001) # Small sleep to prevent high CPU usage

#       End frame

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
