
# --- Imports ---

import cProfile
import cv2
import time
import numpy as np

from recievers.webCamSim import VideoThreadedCapture

from utilities.color_functions import dominant_color, color_offset_calculation, colorTracker, tracker, build_color_LUT
from utilities.screen_alignment_functions import roi_alignment
from utilities.decoding_functions import decode_bitgrid, sync_receiver
from utilities.global_definitions import (
    sender_output_height, sender_output_width,
    roi_window_height, roi_window_width,
    aruco_marker_dictionary, aruco_detector_parameters, aruco_marker_size, aruco_marker_margin,
    display_text_font, display_text_size, display_text_thickness,
    green_bgr, red_bgr
)

# --- Video capture setup ---

videoCapture = VideoThreadedCapture(r"C:\my_projects\optical-laptop-communication\recievers\intervals_test.mp4") # For video test
# videoCapture = VideoThreadedCapture(0) # For live webcam

if not videoCapture.isOpened():
    print("Error: Could not open camera/video.")
    exit()

while True:

    read_was_sucessful, frame = videoCapture.read() # Tries to grab one initial frame to make sure the video capture is "warmed up"

    if read_was_sucessful:
        break

    time.sleep(0.01)

# --- OpenCV window setup ---

cv2.namedWindow("Webcam Receiver", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Webcam Receiver", sender_output_width, sender_output_height)

cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ROI", roi_window_width, roi_window_height)

# --- ArUco detector setup ---

aruco_detector = cv2.aruco.ArucoDetector(aruco_marker_dictionary, aruco_detector_parameters)

# --- Main function ---

def receive_message():

    """
    Receives a message from the sender screen.
    
    Arguments:
        None

    Returns:
        None
    
    """

#   Variable initialization

    bits = ""
    message = ""
    arucos_found = False

    last_color = None

    waiting_for_sync = True
    syncing = False
    interval = 0 # Interval between frames in seconds

    decoding = False
    current_bit_colors = [] # Colors collected for the current bit
    roi_coordinates = None
    frame_bit = 0 # Current frame bit index

    previous_time = time.time()
    frame_count = 0 # Frame count for debugging


    corrected_ranges = {
            "red":    (np.array([0, 100, 100]), np.array([10, 255, 255])),  # hue 0–10
            "red2":   (np.array([160, 100, 100]), np.array([179, 255, 255])),  # hue 160–179
            "green":  (np.array([45, 80, 80]), np.array([75, 255, 255])),
            "blue":   (np.array([95, 120, 70]), np.array([130, 255, 255])),
            "white":  (np.array([0, 0, 220]), np.array([180, 25, 255])),
            "black":  (np.array([0, 0, 0]), np.array([180, 255, 35]))
        }
    
    LUT, color_names = build_color_LUT(corrected_ranges)
    tracker.colors(LUT, color_names)


#   ArUco marker detection

    print("Receiver started — searching for ArUco markers...")

    while True:

        read_was_sucessful, frame = videoCapture.read() # Reads a frame from the video capture

        if not read_was_sucessful:

            print("Error: Failed to capture frame.")
            continue

#       --- Debugging ---

        frame_count += 1

        current_time = time.time()

        if current_time - previous_time >= 1.0:
            print(f"Loops per second: {frame_count}")
            frame_count = 0
            previous_time = current_time

#       --- End of debugging ---

        if arucos_found is False: # If no ArUco markers have been found:

            grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Grayscale the frame

            corners, marker_ids, _ = aruco_detector.detectMarkers(grayscaled_frame) # Call the ArUco detector on the grayscaled frame

            if marker_ids is not None and len(marker_ids) > 0 and roi_coordinates is None: # If markers were detected and there are no ROI coordinates yet:
                roi_coordinates, aruco_marker_side_length, _ = roi_alignment(frame) # Get the ROI coordinates based on the detected markers

#       Display drawings
        
        display = frame.copy() # Create a copy of the frame for display purposes

        if marker_ids is not None and len(marker_ids) > 0:

            cv2.aruco.drawDetectedMarkers(display, corners, marker_ids) # Draw the detected markers on the display frame

            cv2.putText(display, f"{len(marker_ids)} ArUco marker(s) detected", (20, 40), display_text_font, display_text_size, green_bgr, display_text_thickness)
            
            arucos_found = True
            marker_ids = None # Reset marker IDs to avoid repeated processing
            
        else:
            cv2.putText(display, "No ArUco markers detected", (20, 40), display_text_font, display_text_size, red_bgr, display_text_thickness)

        if roi_coordinates is not None and not hasattr(receive_message, "roi_padded"): # If there are ROI coordinates and "recieve_message" doesn't have the attribute "roi_padded":
            
            start_x, end_x, start_y, end_y = roi_coordinates # Unpack the ROI coordinates

#           ROI expansion

            roi_padding_px = (aruco_marker_side_length / aruco_marker_size) * aruco_marker_margin # Calculate the padding in pixels

            start_x = int(start_x - roi_padding_px)
            end_x = int(end_x + roi_padding_px)

            start_y = int(start_y - roi_padding_px)
            end_y = int(end_y + roi_padding_px)

#           Minimized ROI coordinates

            roi_height = end_y - start_y
            roi_width = end_x - start_x

            minimized_start_x = int(start_x - (roi_width / 2))
            minimized_start_y = int(start_y - (roi_height / 2))

            minimized_end_x = int(end_x + (roi_width / 2))
            minimized_end_y = int(end_y + (roi_height / 2))

            receive_message.roi_padded = (start_x, end_x, start_y, end_y) # Assigns the attribute "roi_padded" to "recieve_message" with given values

            if start_x < end_x and start_y < end_y: # If the ROI coordinates are valid:
                cv2.rectangle(display, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
        
        if roi_coordinates is not None: # If there are ROI coordinates:
            roi = frame[start_y:end_y, start_x:end_x] # Extract the ROI from the frame
            minimized_roi = frame[minimized_start_y:minimized_end_y, minimized_start_x:minimized_end_x] # Extract the minimized ROI from the frame
        
        else: # Else (if there aren't any):
            roi = np.zeros((10, 10, 3), dtype = np.uint8) # Create a dummy ROI
            minimized_roi = roi # Set the minimized ROI to the dummy ROI

        minimized_roi = cv2.cvtColor(minimized_roi, cv2.COLOR_BGR2HSV)

        color = dominant_color(minimized_roi) # Get the dominant color in the minimized ROI

        cv2.imshow("ROI", roi)

        cv2.imshow("Webcam Receiver", display)

#       Waiting for sync

        if roi_coordinates is not None:

            # --- Color calibration ---

            if color_calibration:

                corrected_ranges = color_offset_calculation(roi)
                LUT, color_names = build_color_LUT(corrected_ranges)
                tracker.colors(LUT, color_names)
                
                color_calibration = False
                syncing = True

            # --- Sync ---

            elif syncing: # If we're syncing:

                interval, syncing = sync_receiver(minimized_roi, True) # Try to sync and get the interval

            # --- Decode ---

            elif decoding: # If we're decoding:

                recall = False # Initialize recall flag as False
                end_frame = False # Initialize end_frame flag as False
                add_frame = False # Initialize add_frame flag as False

                if color == "blue" and last_color != "blue": # If the color is blue and the last color wasn't blue:
                    
                    end_frame = True
                    add_frame = True

                elif color in ["white", "black"]: # If the color is white or black:

                    add_frame = True

                elif color == "red" and last_color != "red": # If the color is red and the last color wasn't red:

                    recall = True # Set recall to True

                if recall: # If it's a recall frame:
                    message = decode_bitgrid(minimized_roi, frame_bit, add_frame, recall, end_frame) # Decode the bitgrid with recall set to True

                else: # Else (if it's not a recall frame):
                    decode_bitgrid(minimized_roi, frame_bit, add_frame, recall, end_frame)

                if end_frame: # If it's an end frame:
                    frame_bit += 1 # Increment the frame bit index

            last_color = color # Update the last color

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

    if current_bit_colors: # If there are colors collected for the current unfinished bit:
        print(f"Colors collected for last unfinished bit: {current_bit_colors}")

    if bits: # If there are remaining bits not yet converted:
        print(f"Bits not yet converted: {bits}")

    print("Final message:", message)
    print(f"Interval: {interval}s")

    videoCapture.release()
    cv2.destroyAllWindows()

# --- Execution ---

cProfile.run("receive_message()") 