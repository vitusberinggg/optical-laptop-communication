
# --- Imports ---

# Library imports

import cv2
import time
import numpy as np

# Non-library imports

from utilities.encoding_functions import message_to_frame_several_bit_arrays
from utilities.image_generation_functions import (
    render_multicolor_frame, create_color_frame,
    create_color_reference_frame, create_seven_color_reference_frame, create_large_aruco_marker_frame
)

from utilities.global_definitions import (
    message,
    aruco_marker_frame_duration, frame_duration,
    red_bgr, blue_bgr, gray_bgr, orange_bgr,
    sync_colors, number_of_sync_frames, sync_frame_duration, color_map_2bit, color_map_3bit, 
    bits_per_cell, margin, sender_output_width, sender_output_height
)


# --- Helper function ---

def frame_with_margin(frame, margin=15):
    """
    Places `frame` inside a full-screen background with a fixed pixel margin.
    If the frame is too big to fit with the margin, it will be scaled down
    while preserving aspect ratio.
    """
    fh, fw = frame.shape[:2]

    # Compute available area inside the margin
    available_w = sender_output_width - 2 * margin
    available_h = sender_output_height - 2 * margin

    # Determine if scaling is needed
    scale = min(1.0, min(available_w / fw, available_h / fh))  # max scale = 1 (no upscaling)

    new_w = int(fw * scale)
    new_h = int(fh * scale)

    # Resize frame if needed
    if scale < 1.0:
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Create background
    background = np.full((sender_output_height, sender_output_width, 3), gray_bgr, dtype=np.uint8)

    # Compute top-left corner to center the frame
    x0 = (sender_output_width - new_w) // 2
    y0 = (sender_output_height - new_h) // 2

    # Place the frame
    background[y0:y0+new_h, x0:x0+new_w] = frame

    return background




# --- Main function ---

def send_message(message):

    """
    Sends a message by displaying frames on the screen.

    Arguments:
        "message" (str): The message to be sent.

    Returns:
        None
    
    """

    color_reference_frame = create_seven_color_reference_frame()

    sync_frames = []

    for color in sync_colors: # For each color in the sync colors array
        color_frame = create_color_frame(color) # Creates a frame in the color
        sync_frames.append(color_frame) # Adds the color frame to the sync frame list

    frame_bit_arrays = message_to_frame_several_bit_arrays(message, bits_per_cell) # Converts the message to frame bit arrays

    if bits_per_cell == 2:
        color_map = color_map_2bit

    elif bits_per_cell == 3:
        color_map = color_map_3bit

    data_frames = []

    for frame_bit_array in frame_bit_arrays: # For each frame bit array:
        rendered_frame = render_multicolor_frame(frame_bit_array, color_map) # Render the frame
        data_frames.append(rendered_frame) # Add the rendered frame to the list of data frames

    end_frame  = create_color_frame(orange_bgr) # Creates the end frame with the specified color

    # OpenCV window
#   OpenCV window

    window = "SENDER" # The name of the OpenCV window
    cv2.namedWindow(window, cv2.WINDOW_NORMAL) # Creates a window with the specified name
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # Sets the window to fullscreen

    # Aruco marker frames

    aruco_frames = [
    create_large_aruco_marker_frame(position = "right"),
    create_large_aruco_marker_frame(position = "left")
]

    for aruco_frame in aruco_frames:

        aruco_marker_frame_start_time = time.monotonic()

        while time.monotonic() - aruco_marker_frame_start_time < aruco_marker_frame_duration:
            
            cv2.imshow(window, aruco_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                return
            
            time.sleep(0.001)

    # Color reference frame

    color_reference_frame_start_time = time.monotonic()
    color_reference_frame = frame_with_margin(color_reference_frame)

    while time.monotonic() - color_reference_frame_start_time < (4):
        
        cv2.imshow(window, color_reference_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return
        
        time.sleep(0.001)

    # Sync frames

    for _ in range(number_of_sync_frames // 2):

        for sync_frame in sync_frames:
        
            sync_frame_start_time = time.monotonic()
            sync_frame = frame_with_margin(sync_frame)

            while time.monotonic() - sync_frame_start_time < sync_frame_duration:
                
                cv2.imshow(window, sync_frame)
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return
                    
                time.sleep(0.001)

    try:

        # End of sync

        end_of_sync_frame_start_time = time.monotonic()
        end_frame = frame_with_margin(end_frame)

        while time.monotonic() - end_of_sync_frame_start_time < (frame_duration/2):
            
            cv2.imshow(window, end_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                return
            
            time.sleep(0.001)

        # Data transfer loop

        for frame in data_frames: # For each frame:
            
            frame_start_time = time.monotonic() # Records the start time for the current frame
            frame = frame_with_margin(frame)

            while time.monotonic() - frame_start_time < frame_duration: # While the frame duration limit hasn't been reached:

                cv2.imshow(window, frame) # Display the current frame in the window

                if cv2.waitKey(1) & 0xFF == ord("q"): # If "Q" is pressed:
                    return # Exit the function
                
                time.sleep(0.001) # Small sleep to prevent high CPU usage

        # End frame

        end_frame_start_time = time.monotonic() # Records the start time for the end frame
        end_frame = frame_with_margin(end_frame)

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