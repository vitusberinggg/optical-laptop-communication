
# --- Imports ---

import cv2
import numpy as np
import time

from utilities.generate_reference_image import generate_reference_image
from utilities.message_to_frame_bit_arrays import message_to_frame_bit_arrays
from utilities.render_frame import render_frame

from utilities.global_definitions import (
    sender_output_height, sender_output_width,
    reference_image_duration,
    frame_duration)

# ---- Definitions ----

message = "HELLO"

fps = 30

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

def send_message(message = "HELLO"):

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
    black_frame = create_color_frame((0, 0, 0))

    win = "SENDER"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL) # Creates a window with the specified name
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # Sets the window to fullscreen

    start_time = time.time()

    while time.time() - start_time < reference_image_duration:

        cv2.imshow(win, reference_image_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return

    try:

        while True:

            for frame in data_frames:

                frame_start = time.time()

                while time.time() - frame_start < frame_duration:

                    cv2.imshow(win, frame)

                    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                        return

            end_start = time.time()

            while time.time() - end_start < 0.3:
                cv2.imshow(win, end_frame)

                if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                    return

            black_start = time.time()

            while time.time() - black_start < 0.5:
                cv2.imshow(win, black_frame)

                if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                    return

    except KeyboardInterrupt:
        pass

    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    send_message(message)
