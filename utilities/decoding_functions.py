
# --- Imports ---

from collections import Counter
import numpy as np
import cv2

from utilities import color_functions
from utilities.global_definitions import number_of_rows, number_of_columns, number_of_sync_frames

# --- Definitions ---

bits = [[[]]] # 3D list to hold the bits "decode_bitgrid" returns (bits[frame][row][column])

# --- Functions ---

def decode_bitgrid(frame, frame_bit = 0, add_frame = False, recall = False, end_frame = False):

    """
    Decodes a grid of bits.

    Arguments:
        "frame"
        "frame_bit"
        "add_frame"
        "recall"
        "end_frame"

    Returns:
        None

    """

    global bits

    frame_height, frame_width = frame.shape[:2]
    bit_cell_height = frame_height / number_of_rows
    bit_cell_width  = frame_width / number_of_columns

    # --- ADD FRAME ---
    if add_frame:

        # ensure list for this frame
        while len(bits) <= frame_bit:
            bits.append([])

        for row in range(number_of_rows):

            # ensure row list exists
            while len(bits[frame_bit]) <= row:
                bits[frame_bit].append([])

            for column in range(number_of_columns):

                # ensure column exists
                while len(bits[frame_bit][row]) <= column:
                    bits[frame_bit][row].append(None)

                # extract ROI
                y0 = int(row * bit_cell_height)
                y1 = int(y0 + bit_cell_height)
                x0 = int(column * bit_cell_width)
                x1 = int(x0 + bit_cell_width)
                cell = frame[y0:y1, x0:x1]

                if end_frame:
                    bit = color_functions.tracker.end_bit(row, column)

                    # Ensure safe bit (string "0" / "1")
                    if bit not in ["0", "1"]:
                        bit = "0"

                    bits[frame_bit][row][column] = bit

                else:
                    color_functions.tracker.add_frame(cell, row, column)

        return None

    # --- RECALL AND DECODE ---
    if recall:

        collected_bytes = []
        current_byte = []

        for f in range(frame_bit):      # each finalized frame
            for row in range(number_of_rows):
                for column in range(number_of_columns):

                    value = bits[f][row][column]

                    # safety: convert None → "0"
                    if value is None:
                        value = "0"

                    current_byte.append(value)

                    if len(current_byte) == 8:
                        collected_bytes.append(current_byte)
                        current_byte = []

        print(f"Decoded {len(collected_bytes)} bytes from {frame_bit} frames.")
        for i, byte_bits in enumerate(collected_bytes):
            byte_str = "".join(str(b) for b in byte_bits)
            print(f"Byte {i}: {byte_str} (char: '{chr(int(byte_str,2))}')")
        return bits_to_message(collected_bytes)

    return None

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

import time

import time

def sync_receiver(roi, verbose=True, state={}):
    """
    Syncs timing by detecting black/white transitions.

    This version is designed for cases where YOU provide each ROI frame
    manually

    Args:
        roi (np.ndarray): The ROI frame for this moment.
        detect_color_fn: Function that returns "black" or "white".
        transitions_needed (int): How many transitions we need to detect.
        verbose (bool): Print debug info.
        state (dict): Internal persistent state across calls.

    Returns:
        float | None:
            - Returns frame_interval (float) when enough transitions are detected.
            - Returns None otherwise.
    """

    # --- INITIALIZE STATE ---
    if "last_color" not in state:
        state["last_color"] = None
        state["transition_times"] = []
        if verbose:
            print("[SYNC] Waiting for first stable color...")

    # --- Detect color ---
    color = color_functions.dominant_color(roi)

    # First color → just store it
    if state["last_color"] is None:
        state["last_color"] = color
        if verbose:
            print(f"[SYNC] Initial color = {color}")
        return 0, True

    # --- Detect transition ---
    if color != state["last_color"]:
        timestamp = time.time()
        state["transition_times"].append(timestamp)

        if verbose:
            t = len(state["transition_times"])
            print(f"[SYNC] Transition {t}: {state['last_color']} → {color}")

        state["last_color"] = color

        # --- Enough transitions? Compute interval ---
        if len(state["transition_times"]) >= number_of_sync_frames:
            times = state["transition_times"]
            diffs = [
                times[i+1] - times[i]
                for i in range(len(times)-1)
            ]
            frame_interval = sum(diffs) / len(diffs)

            if verbose:
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
