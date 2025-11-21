
# --- Imports ---

import cv2 # Imports the OpenCV library
import time

from utilities.image_generation_functions import render_frame, create_color_frame, create_aruco_marker_frame, create_color_reference_frame
from utilities.encoding_functions import message_to_frame_bit_arrays

from utilities.global_definitions import (
    aruco_marker_frame_duration,
    frame_duration,
    blue_bgr, 
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

    color_reference_frame = create_color_reference_frame() 

    preamble_frames = []

    for color in preamble_colors: # For each color in the preamble colors array
        color_frame = create_color_frame(color) # Creates a frame in the color
        preamble_frames.append(color_frame) # Adds the color frame to the preamble frame list

    frame_bit_arrays = message_to_frame_bit_arrays(message) # Converts the message to frame bit arrays

    data_frames = []

    for frame_bit_array in frame_bit_arrays: # For each frame bit array:
        rendered_frame = render_frame(frame_bit_array) # Render the frame
        data_frames.append(rendered_frame) # Add the rendered frame to the list of data frames

    sync_frame = create_color_frame(blue_bgr)
    end_frame  = create_color_frame(end_frame_color) # Creates the end frame with the specified color

    window = "SENDER" # The name of the OpenCV window
    cv2.namedWindow(window, cv2.WINDOW_NORMAL) # Creates a window with the specified name
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # Sets the window to fullscreen

#   Aruco marker frame

    # --- ArUco marker frames (right first, then left) ---
    aruco_frames = [
        create_aruco_marker_frame(position="right"),
        create_aruco_marker_frame(position="left")
    ]

    for aruco_frame in aruco_frames:
        start_time = time.monotonic()
        while time.monotonic() - start_time < aruco_marker_frame_duration:
            cv2.imshow(window, aruco_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                return
            time.sleep(0.001)

    color_start_time = time.monotonic()
    while time.monotonic() - color_start_time < frame_duration:
        cv2.imshow(window, color_reference_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return
        time.sleep(0.001)

    try:

#       Data transfer loop

        for frame in data_frames: # For each frame:

            frame_start_time = time.monotonic() # Records the start time for the current frame

            while time.monotonic() - frame_start_time < frame_duration: # While the frame duration limit hasn't been reached:

                cv2.imshow(window, frame) # Display the current frame in the window

                if cv2.waitKey(1) & 0xFF == ord("q"): # If "Q" is pressed:
                    return # Exit the function
                
                time.sleep(0.001) # Small sleep to prevent high CPU usage

            sync_frame_start_time = time.monotonic()

            while time.monotonic() - sync_frame_start_time < frame_duration:

                cv2.imshow(window, sync_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"): # If "Q" is pressed:
                    return # Exit the function
                
                time.sleep(0.001) # Small sleep to prevent high CPU usage

#       End frame

        end_frame_start_time = time.monotonic() # Records the start time for the end frame

        while time.monotonic() - end_frame_start_time < frame_duration: # While the end frame duration limit hasn't been reached:

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
