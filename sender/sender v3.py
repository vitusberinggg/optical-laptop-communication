
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

def message_to_frame_arrays(message):

    """
    Turns a message into a list of frames represented as 2D NumPy arrays of bits.

    Arguments:
        "message" (str): The message to be converted.

    Returns:
        "frame_arrays": A list of 2D NumPy arrays representing the frames.
    
    """

    binary_list = []

    for character in message: # For each character in the message:
        ascii_value = ord(character) # Convert it to ASCII
        binary_string = format(ascii_value, "08b") # Format the value as an 8-bit binary string
        binary_list.append(binary_string) # Add the string to the binary list
    
    bits = "".join(binary_list) # Merge all strings in the binary list into a single string

    frame_capacity = rows * columns
    frame_arrays = []

    for start_index in range(0, len(bits), frame_capacity): # For each starting index in the range 0 - len(bits), stepping with the frame capacity:

        chunk = bits[start_index:start_index + frame_capacity] # Slice a chunk the size of the frame capacity from the string of bits

        if len(chunk) < frame_capacity: # If the length of the chunk is smaller than the frame capacity
            chunk = chunk.ljust(frame_capacity, "0") # Pad with 0's until the chunk is the same size as the frame capacity

        frame_array = np.array(list(chunk), dtype = np.uint8).reshape((rows, columns)) # Convert the chunk to a list, convert the list into an NumPy array (more efficient) and then reshape the one-dimensional array into a 2D array

        frame_arrays.append(frame_array) # Add the frame array into the list of frames

    return frame_arrays

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

# --- Main function ---

def show_sequence(message="HELLO"):
    # generate reference image and convert to BGR for display
    ref = generate_reference_image()
    ref_bgr = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)

    # prepare frames
    bit_frames = message_to_frame_arrays_bits(message)
    data_frames = [render_frame(f) for f in bit_frames]

    SYNC_FRAME = create_color_frame((0,255,0))   # green indicates start-of-data
    END_FRAME  = create_color_frame((0,0,255))   # red indicates end-of-message
    BLACK_FRAME = create_color_frame((0,0,0))

    win = "SENDER"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Sender: showing reference. Position receiver and press 'q' to stop.")
    # show reference for reference_image_duration
    start = time.time()
    while time.time() - start < reference_image_duration:
        cv2.imshow(win, ref_bgr)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return

    # Now transmit in a loop
    try:
        while True:
            # SYNC
            for _ in range(int(fps * 0.3)):  # short green pulse
                cv2.imshow(win, SYNC_FRAME)
                if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

            # send each data frame for bit_time seconds
            for df in data_frames:
                samples = max(1, int(bit_time * fps))
                for _ in range(samples):
                    cv2.imshow(win, df)
                    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                        raise KeyboardInterrupt

            # END pulse
            for _ in range(int(fps * 0.3)):
                cv2.imshow(win, END_FRAME)
                if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

            # short gap
            for _ in range(int(fps * 0.5)):
                cv2.imshow(win, BLACK_FRAME)
                if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # change message here
    show_sequence("HELLO FROM SENDER! THIS IS ECC-BASED LINK.")
