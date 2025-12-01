
# --- Imports ---
import numpy as np
import time
from utilities import color_functions

from utilities.global_definitions import number_of_rows, number_of_columns, number_of_sync_frames
from utilities.decoding_functions_v3_1 import bits_to_message

# --- Functions ---

bits = [[[]]]

def decode_bitgrid(frame, frame_bit = 0, add_frame = False, recall = False, end_frame = False):

    """
    
    """
    
    global bits

    h, w = frame.shape[:2]
    bit_cell_height = h / number_of_rows
    bit_cell_width  = w / number_of_columns

    if add_frame:

        while len(bits) <= frame_bit:
            bits.append([])

        for row in range(number_of_rows):

            while len(bits[frame_bit]) <= row:
                bits[frame_bit].append([])

            for column in range(number_of_columns):

                while len(bits[frame_bit][row]) <= column:
                    bits[frame_bit][row].append(None)

                y0 = int(row * bit_cell_height)
                y1 = int(y0 + bit_cell_height)
                x0 = int(column * bit_cell_width)
                x1 = int(x0 + bit_cell_width)
                cell = frame[y0:y1, x0:x1]

                if end_frame:
                    bit = color_functions.tracker.end_bit(row, column)

                    if bit not in ["0", "1"]:
                        bit = "0"

                    bits[frame_bit][row][column] = bit

                else:
                    color_functions.tracker.add_frame(cell, row, column)

        return None

    if recall:

        collected_bytes = []
        current_byte = []

        for f in range(frame_bit):
            for row in range(number_of_rows):
                for column in range(number_of_columns):

                    value = bits[f][row][column]

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

def sync_interval_detector(color, printing = True, sync_state_dictionary = {}):

    """
    Syncs timing by detecting black/white transitions.

    Args:
        "color" (str):
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

        # Interval calculation

        if amount_of_timestamps >= (number_of_sync_frames - 1): # If the amount of timestamps is equal to or more than the amount of sync frames - 1 (the amount of transitions):

            timestamps = sync_state_dictionary["transition_timestamps"] # Get the list of timestamps

            frame_intervals = [] # Create an empty list for timestamp differences

            for timestamp_index in range (len(timestamps) - 1): # For each timestamp index in the list of timestamps:
                frame_intervals.append(timestamps[timestamp_index + 1] - timestamps[timestamp_index]) # Add the difference between that timestamp and the next one to the timestamp differences list

            #average_frame_interval = sum(frame_intervals) / len(frame_intervals) # Calculate the average frame interval
            median_interval = float(np.median(frame_intervals))

            if printing:
                print(f"[SYNC] Estimated frame interval: {median_interval:.4f} seconds")

            return median_interval, False

    return 0, True # If "color" = "last_color", quit (no transition detected yet)
