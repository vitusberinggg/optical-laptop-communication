
 # --- Imports ---

import cv2
import numpy as np
import time
import math

# ---- Definitions ----

OUT_W = 800
OUT_H = 600

reference_image_seed = 42
reference_image_duration = 2.0

fps = 15

columns = 1
rows = 1

bit_time = 0.5

bit_cell_width = OUT_W // columns
bit_cell_height = OUT_H // rows

# --- Helper functions ---

def generate_reference_image():

    """
    Generates the reference image by putting the image seed into a bit generator.

    Arguments:
        None

    Returns:
        "reference_image": The reference image.

    """

    bit_generator = np.random.RandomState(reference_image_seed)
    reference_image = bit_generator.randint(0, 256, (OUT_H, OUT_W), dtype = np.uint8)

    return reference_image

def message_to_frame_bit_arrays(message):

    """
    Turns a message into a list of frames represented as 2D NumPy arrays of bits.

    Arguments:
        "message" (str): The message to be converted.

    Returns:
        "frame_bit_arrays": A list of 2D NumPy arrays representing the frames.
    
    """

    binary_list = []

    for character in message: # For each character in the message:
        ascii_value = ord(character) # Convert it to ASCII
        binary_string = format(ascii_value, "08b") # Format the value as an 8-bit binary string
        binary_list.append(binary_string) # Add the string to the binary list
    
    bits = "".join(binary_list) # Merge all strings in the binary list into a single string

    frame_capacity = rows * columns
    frame_bit_arrays = []

    for start_index in range(0, len(bits), frame_capacity): # For each starting index in the range 0 - len(bits), stepping with the frame capacity:

        chunk = bits[start_index:start_index + frame_capacity] # Slice a chunk the size of the frame capacity from the string of bits

        if len(chunk) < frame_capacity: # If the length of the chunk is smaller than the frame capacity
            chunk = chunk.ljust(frame_capacity, "0") # Pad with 0's until the chunk is the same size as the frame capacity

        frame_array = np.array(list(chunk), dtype = np.uint8).reshape((rows, columns)) # Convert the chunk to a list, convert the list into an NumPy array (more efficient) and then reshape the one-dimensional array into a 2D array

        frame_bit_arrays.append(frame_array) # Add the frame array into the list of frames

    return frame_bit_arrays

def render_frame(bitgrid):

    """
    Renders a frame from a NumPy array.

    Arguments:
        "bitgrid" (np.ndarray): A 2D NumPy array of bits.

    Returns:
        "image" (np.ndarray): A NumPy array representing the image frame pixels.

    """

    image = np.zeros((OUT_H, OUT_W, 3), dtype = np.uint8)

    for row in range(rows): # For each row:

        for column in range(columns): # For each column:

            bit = int(bitgrid[row, column]) # Get the bit at the current position

            if bit == 1: # If the bit is 1:
                color = (255, 255, 255) # Set the color to white
            
            else: # Else (if the bit is 0):
                color = (0, 0, 0) # Set the color to black

            start_x_coordinate = column * bit_cell_width
            end_x_coordinate = start_x_coordinate + bit_cell_width

            start_y_coordinate = row * bit_cell_height
            end_y_coordinate = start_y_coordinate + bit_cell_height

            cv2.rectangle(image, (start_x_coordinate, start_y_coordinate), (end_x_coordinate - 1, end_y_coordinate - 1), color, thickness = -1) # Draw the rectangle on the image

    return image

def create_color_frame(color):

    """
    Creates a solid color frame.

    Arguments:
        "color" (tuple): A tuple representing the BGR color.

    Returns:
        "frame" (np.ndarray): A NumPy array representing the solid color frame pixels.

    """

    return np.full((OUT_H, OUT_W, 3), color, dtype = np.uint8)

def show_and_wait(win, frame, delay_ms):

    """
    Shows a frame in a window and waits.

    Arguments:
        "win" (str): The window name.
        "frame" (np.ndarray): The frame to be shown.
        "delay_ms" (int): The delay in milliseconds.
    
    Returns:
        
    
    """

    cv2.imshow(win, frame)
    key = cv2.waitKey(delay_ms) & 0xFF # Waits for a key press for the specified delay
    if key == ord('q') or key == 27:
        return True
    
    return False

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

    sync_frame = create_color_frame((0, 255, 0))
    end_frame  = create_color_frame((0, 0, 255))
    black_frame = create_color_frame((0, 0, 0))

    win = "SENDER"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL) # Creates a window with the specified name
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # Sets the window to fullscreen

    try:

        start_time = time.time()
        delay_ms = max(1, int(1000 / fps))

        while time.time() - start_time < reference_image_duration: # While the elapsed time is less than the reference image duration:

            if show_and_wait(win, reference_image_bgr, delay_ms):
                return

        while True:

            sync_duration = 0.3
            sync_start = time.time()
            while time.time() - sync_start < sync_duration:
                if show_and_wait(win, sync_frame, delay_ms):
                    return

            samples_per_bit = max(1, int(bit_time * fps))
            for frame in data_frames:
                for _ in range(samples_per_bit):
                    if show_and_wait(win, frame, delay_ms):
                        return

            end_duration = 0.3
            end_start = time.time()
            while time.time() - end_start < end_duration:
                if show_and_wait(win, end_frame, delay_ms):
                    return

            black_duration = 0.5
            black_start = time.time()
            while time.time() - black_start < black_duration:
                if show_and_wait(win, black_frame, delay_ms):
                    return

    except KeyboardInterrupt:
        pass

    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    send_message("HELLO")