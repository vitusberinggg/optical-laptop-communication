
# --- Imports ---

import cv2
import numpy as np
import time

# --- Definitions ---

screen_width = 2650
screen_height = 1440

marker_size = 120
frame_padding = 120

bit_time = 0.35
fps = 30

# --- ArUco marker setup ---

arUco_dictionnary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50) # Gets the 4 x 4 ArUco dictionary with 50 unique markers
marker_ids = [0, 1, 2, 3] # IDs for the four corner markers

# --- Helper functions ---

def create_aruco_marker(marker_id):

    """
    Creates a BGR ArUco marker image with the specified ID and size.
    
    Arguments:
        "marker_id": The ID of the marker to create.
        "size": The size of the marker in pixels.

    Returns:
        "marker_bgr": A numpy array representing the marker image in BGR format.

    """

    marker = np.zeros((marker_size, marker_size), dtype = np.uint8) # Creates a blank image for the marker

    cv2.aruco.generateImageMarker(arUco_dictionnary, marker_id, marker_size, marker, 1) # Generates the marker image with the specified ID and size

    marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR) # Converts the marker to BGR format
    
    return marker_bgr

def create_frame(center_color=(0, 0, 0), margin_color=(128, 128, 128), margin=50):
    """
    Creates a frame with ArUco markers in the corners and a smaller colored center square.
    
    Arguments:
        center_color: BGR color for the center square.
        margin_color: BGR color for the rest of the frame (background/margin).
        margin: Distance from the edges to keep markers clear.
    """
    # Fill the frame with gray instead of black
    frame = np.full((screen_height, screen_width, 3), margin_color, dtype=np.uint8)

    # Coordinates for ArUco markers
    marker_coordinates = [
        (10, 10),  # Top-left
        (screen_width - 10 - marker_size, 10),  # Top-right
        (screen_width - 10 - marker_size, screen_height - 10 - marker_size),  # Bottom-right
        (10, screen_height - 10 - marker_size),  # Bottom-left
    ]

    # Place ArUco markers
    for idx, (x, y) in enumerate(marker_coordinates):
        marker = create_aruco_marker(marker_ids[idx])
        frame[y:y+marker_size, x:x+marker_size] = marker

    # Smaller center square
    center_x_start = margin + marker_size
    center_x_end = screen_width - margin - marker_size
    center_y_start = margin + marker_size
    center_y_end = screen_height - margin - marker_size
    cv2.rectangle(frame, (center_x_start, center_y_start), (center_x_end, center_y_end), center_color, thickness=-1)

    return frame

def message_to_frames(message, bit_time = 0.35, fps = 15):

    """
    Converts a message string into a sequence of frames for transmission.
    
    Arguments:
        "message": The message string to convert.
        "bit_time": Duration of each bit in seconds.
        "fps": Frames per second for the transmission.
    
    Returns:
        "frames": A list of frames representing the message.

    """

    frames = []

    samples_per_bit = max(1, int(bit_time * fps))

    f = create_frame(center_color = (0, 255, 0))
    frames += [f] * samples_per_bit * 3 

    for character in message:
        bits = format(ord(character), "08b")

        for bit in bits:

            if bit == "1":
                color = (255, 255, 255)

            else:
                color = (0, 0, 0)

            f = create_frame(center_color = color)
            frames += [f] * samples_per_bit

            f = create_frame(center_color = (255, 0, 0))
            frames += [f] * samples_per_bit

        f = create_frame(center_color = (0, 0, 255))
            frames += [f] * samples_per_bit

    return frames

def show_message(message):

    """
    Displays the message on the screen using OpenCV.
    
    Arguments:
        "message": The message string to display.
        
    Returns:
        None

    """

    fps = 15
    bit_time = 0.35
    frames = message_to_frames(message, bit_time=bit_time, fps=fps)

    win = "SENDER"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Position receiver so webcam sees this display. Press q to quit.")
    idx = 0
    while True:
        cv2.imshow(win, frames[idx])
        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        idx += 1
        if idx >= len(frames):
            idx = 0
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_message("HELLO")
