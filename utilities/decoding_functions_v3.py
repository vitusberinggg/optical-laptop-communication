
# --- Imports ---

import numpy as np
import time

from utilities import color_functions_v3
from utilities.color_functions_v3 import dominant_color
from utilities.global_definitions import number_of_sync_frames

# --- Definitions ---

bitgrids = []

# --- Functions ---

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

def bits_to_message(byte_matrix):

    """
    Converts a 2D list of bits (each inner list is a byte) into a readable message.

    Arguments:
        bit_matrix (list of list of int): Each inner list should contain 8 bits (0s or 1s).

    Returns:
        str: The decoded message as a string.

    """

    characters = []

    for byte_bits in byte_matrix:

        s = "".join(['1' if b else '0' for b in byte_bits])

        try:
            characters.append(chr(int(s, 2)))

        except ValueError:
            characters.append('?')  # placeholder for invalid or partial bytes
            
    return "".join(characters)

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

        # Interval calculation

        if amount_of_timestamps >= (number_of_sync_frames - 1): # If the amount of timestamps is equal to or more than the amount of sync frames - 1 (the amount of transitions):

            timestamps = sync_state_dictionary["transition_timestamps"] # Get the list of timestamps

            frame_intervals = [] # Create an empty list for timestamp differences

            for timestamp_index in range (len(timestamps) - 1): # For each timestamp index in the list of timestamps:
                frame_intervals.append(timestamps[timestamp_index + 1] - timestamps[timestamp_index]) # Add the difference between that timestamp and the next one to the timestamp differences list

            average_frame_interval = sum(frame_intervals) / len(frame_intervals) # Calculate the average frame interval

            if printing:
                print(f"[SYNC] Estimated frame interval: {average_frame_interval:.4f} seconds")

            return average_frame_interval, False

    return 0, True # If "color" = "last_color", quit (no transition detected yet)