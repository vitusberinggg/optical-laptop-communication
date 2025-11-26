
# --- Imports ---

from collections import Counter
import numpy as np
import cv2
import time

from utilities import color_functions_v3
from utilities.color_functions_v3 import dominant_color
from utilities.global_definitions import number_of_rows, number_of_columns, number_of_sync_frames

# --- Definitions ---

bitgrids = []    # stores each (rows × cols) bitgrid individually

# --- Functions ---


# --- Main Decoder Function ---

def decode_bitgrid(hsv_frame, add_frame=False, recall=False, end_frame=False):
    """
    Handles bitgrid collection and decoding.

    Args:
        hsv_frame: HSV frame for processing (only used when add_frame=True)
        add_frame: Add this frame to the tracker
        recall: Decode collected bitgrids into bytes and characters
        end_frame: Marks the end of the bit period (pushes 1 full bitgrid)

    Returns:
        str | None: Decoded message (if recall=True)
    """
    global bitgrids

    # --- ADDING FRAMES ---
    if add_frame:
        if end_frame:
            # Retrieve completed bitgrid from tracker
            bitgrid = color_functions_v3.tracker.end_bit()   # e.g. shape (8, 16)

            if bitgrid is not None:
                bitgrids.append(bitgrid)   # Store safely as a separate frame
                print(f"bitgrids: {bitgrids}")

            color_functions_v3.tracker.reset()
        else:
            color_functions_v3.tracker.add_frame(hsv_frame)

        return None


    # --- DECODING ---
    if recall:
        if len(bitgrids) == 0:
            print("No bitgrids collected yet.")
            return None

        # Combine all bitgrids horizontally
        combined = np.hstack(bitgrids)     # shape becomes (8, N)

        flat = combined.ravel()
        num_bytes = len(flat) // 8

        # Split into 8-bit chunks
        byte_matrix = flat[:num_bytes * 8].reshape(-1, 8)

        print(f"Decoded {len(byte_matrix)} bytes:")

        for i, byte_bits in enumerate(byte_matrix):
            # Convert booleans to '0'/'1'
            s = "".join(['1' if b else '0' for b in byte_bits])
            try:
                char = chr(int(s, 2))
            except ValueError:
                char = '?'
            print(f"Byte {i}: {s} (char: '{char}')")

        return bits_to_message(byte_matrix)

    return None



# --- Helper: Convert 8-bit arrays to characters ---

def bits_to_message(byte_matrix):
    chars = []
    for byte_bits in byte_matrix:
        s = "".join(['1' if b else '0' for b in byte_bits])
        try:
            chars.append(chr(int(s, 2)))
        except ValueError:
            chars.append('?')  # placeholder for invalid or partial bytes
    return "".join(chars)

'''
def bits_to_message(bit_matrix):
    """
    Converts a 2D list of bits (each inner list is a byte) into a readable message.

    Arguments:
        bit_matrix (list of list of int): Each inner list should contain 8 bits (0s or 1s).

    Returns:
        str: The decoded message as a string.
    """
    characters = []

    for byte_bits in bit_matrix:
        if len(byte_bits) != 8:
            continue  # skip invalid bytes
        byte_str = "".join(str(b) for b in byte_bits)
        characters.append(chr(int(byte_str, 2)))

    return "".join(characters)
'''

def sync_interval_detector(roi, printing = True, sync_state_dictionary = {}):

    """
    Syncs timing by detecting black/white transitions.

    Args:
        "roi" (np.ndarray): The ROI frame.
        "detect_color_fn": Function that returns "black" or "white".
        "transitions_needed" (int): How many transitions we need to detect.
        "printing" (bool): Print debug info.
        "sync_state_dictionary" (dict): Internal persistent state across calls.

    Returns:
        "frame_interval" (float):

    """

    # Sync state dictionary initialization

    if "last_color" not in sync_state_dictionary:

        sync_state_dictionary["last_color"] = None
        sync_state_dictionary["transition_timestamps"] = []

        if printing:
            print("[SYNC] Initialized sync state dictionary, waiting for first stable color...")

    color = dominant_color(roi) # Gets the dominant color in the current ROI

    # First function call

    if sync_state_dictionary["last_color"] is None: # If this is the first function call:
        sync_state_dictionary["last_color"] = color # Store the current color in "last_color"

        if printing:
            print(f"[SYNC] Initial color = {color}")

        return 0, True # Quit the function early (no transition has occured yet)

    # Transition detection

    if color != sync_state_dictionary["last_color"]: # If the current color isn't the same as the last color:

        timestamp = time.time() # Capture the time
        sync_state_dictionary["transition_timestamps"].append(timestamp) # Save the timestamp in the sync state dictionary
        amount_of_timestamps = len(sync_state_dictionary["transition_timestamps"])

        if printing:
            print(f"[SYNC] Transition {amount_of_timestamps}: {sync_state_dictionary['last_color']} → {color}")

        sync_state_dictionary["last_color"] = color # Update "last_color"

        if amount_of_timestamps >= (number_of_sync_frames - 1):

            times = sync_state_dictionary["transition_timestamps"]

            diffs = []

            for i in range (len(times) - 1):
                diffs.append(times[i + 1] - times[i])

            frame_interval = sum(diffs) / len(diffs)

            if printing:
                print(f"[SYNC] Estimated frame interval: {frame_interval:.4f} seconds")

            return frame_interval, False

    return 0, True



def decode_bits_with_blue(frames, roi_size=100, verbose=False):
    """
    Decodes bits from a sequence of frames where each bit is a colored frame:
    - white: bit 1
    - black: bit 0
    - blue: separator (next bit ready)
    
    Arguments:
        frames (list of np.ndarray): List of frames (BGR) from the sender.
        roi_size (int): Size of the square region to read the color from.
        verbose (bool): Print debug info.
    
    Returns:
        str: Binary string representing the decoded bits.
    """
    def read_color(frame):
        """Detects dominant color in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (0,0,200), (180,30,255))
        black_mask = cv2.inRange(hsv, (0,0,0), (180,255,50))
        blue_mask  = cv2.inRange(hsv, (100,150,0), (140,255,255))
        green_mask = cv2.inRange(hsv, (45,80,80), (75,255,255))
        counts = {
            "white": int(cv2.countNonZero(white_mask)),
            "black": int(cv2.countNonZero(black_mask)),
            "blue": int(cv2.countNonZero(blue_mask)),
            "green": int(cv2.countNonZero(green_mask)),
        }
        return max(counts, key=counts.get)

    bits = ""
    last_color = None
    bit_ready = True  # ready for first bit

    for frame in frames:
        h, w = frame.shape[:2]
        cx, cy = w//2, h//2
        roi = frame[max(0,cy-roi_size):min(h,cy+roi_size), max(0,cx-roi_size):min(w,cx+roi_size)]
        color = read_color(roi)

        if color in ["blue", "green"]:
            bit_ready = True
            if verbose:
                print("Blue separator detected — next bit ready.")
        elif color in ["white","black"] and bit_ready:
            bits += "1" if color == "white" else "0"
            if verbose:
                print(f"Decoded bit: {bits[-1]}")
            bit_ready = False  # wait for next blue separator

        last_color = color

    return bits
