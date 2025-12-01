

# --- Imports ---

import cProfile
import cv2
import time
import numpy as np

from webcam_simulation.webcamSimulator import VideoThreadedCapture

from utilities.color_functions import dominant_color
from utilities.color_functions_v3_1 import color_offset_calculation, tracker, build_color_LUT
from utilities.screen_alignment_functions import roi_alignment_for_large_markers
#from utilities.decoding_functions_v3_1 import sync_interval_detector
from utilities.decoding_functions import decode_bitgrid, sync_interval_detector
from utilities.global_definitions import (
    laptop_webcam_pixel_height, laptop_webcam_pixel_width,
    sender_output_height, sender_output_width,
    roi_window_height, roi_window_width,
    aruco_marker_dictionary, aruco_detector_parameters, aruco_marker_size, aruco_marker_margin,
    display_text_font, display_text_size, display_text_thickness,
    green_bgr, red_bgr, yellow_bgr,
    roi_rectangle_thickness, minimized_roi_rectangle_thickness
)

# --- Video capture setup ---

videoCapture = VideoThreadedCapture(r"C:\Users\eanpaln\Videos\Screen Recordings\rec2.mp4") # For video test
#videoCapture = cv2.VideoCapture(0, cv2.CAP_DSHOW) # For live webcam
"""
# Resolution

videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, laptop_webcam_pixel_width)
videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, laptop_webcam_pixel_height)

# White balance

videoCapture.set(cv2.CAP_PROP_AUTO_WB, 0) # Disables auto white balance
videoCapture.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 3000)
print(f"\n[INFO] Video capture white balance: {videoCapture.get(cv2.CAP_PROP_WB_TEMPERATURE)}")

# Exposure

videoCapture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # Disables auto exposure
videoCapture.set(cv2.CAP_PROP_EXPOSURE, -5) # Lower value = Darker
print(f"\n[INFO] Video capture exposure: {videoCapture.get(cv2.CAP_PROP_EXPOSURE)}")

# Gain

videoCapture.set(cv2.CAP_PROP_GAIN, 0) # Disables auto gain


if not videoCapture.isOpened():
    print("\n[WARNING] Couldn't start video capture.")
    exit()
"""
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

    # Variable initialization

    bits = ""
    message = ""

    minimized_roi_fraction = 1/5

    marker_ids = None
    corners = None

    last_color = None

    last_frame_time = None 

    frame_bit = 0 

    interval = 0 # Interval between frames in seconds

    current_bit_colors = [] # Colors collected for the current bit
    roi_coordinates = None

    has_printed_aruco_detector_message = False
    has_printed_decoding_message = False

    current_state = "aruco_marker_detection"

    # --- Debugging ---

    """

    previous_time = time.time()
    frame_count = 0

    

    # --- End of debugging ---

    print("\n[INFO] Receiver started")

    actual_capture_width = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_capture_height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(f"\n[INFO] Video capture resolution: {round(actual_capture_width)} x {round(actual_capture_height)}")
   
    # --- Debugging ---

    

    print(f"[DEBUGGING] ArUco marker dictionary: {type(aruco_marker_dictionary)}")
    print(f"[DEBUGGING] ArUco detector parameters: {type(aruco_detector_parameters)}")   

    """

    # --- End of debugging ---

    try:

        while True:

           read_was_sucessful, frame = videoCapture.read()  # Reads a frame from the video capture

           if not read_was_sucessful:
               print("\n[INFO] End of video file reached.")
               break


            # ArUco marker detection

           if current_state == "aruco_marker_detection": # If no ArUco markers have been found:

                try:
                    
                    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Grayscale the frame

                    if not has_printed_aruco_detector_message: # If we haven't already printed the ArUco detector message:
                        print("\n[INFO] Running the ArUco marker detector...")
                        has_printed_aruco_detector_message = True

                    corners, marker_ids, _ = aruco_detector.detectMarkers(grayscaled_frame) # Call the ArUco detector on the grayscaled frame

                    if marker_ids is not None and corners is not None and len(marker_ids) > 0 and roi_coordinates is None: # If markers were detected and there are no ROI coordinates yet:
                        roi_coordinates, aruco_marker_side_length, _ = roi_alignment_for_large_markers(corners, marker_ids, frame) # Get the ROI coordinates based on the detected markers
                    
                except Exception:
                    print("\n[WARNING] ArUco detection failed.")
                    aruco_marker_side_length = 0

            # Display drawings
            
           display = frame.copy() # Create a copy of the frame for display purposes

           if marker_ids is not None and len(marker_ids) > 0:

                cv2.aruco.drawDetectedMarkers(display, corners, marker_ids) # Draw the detected markers on the display frame

                cv2.putText(display, f"{len(marker_ids)} ArUco marker(s) detected", (20, 40), display_text_font, display_text_size, green_bgr, display_text_thickness)
            
                # marker_ids = None # Reset marker IDs to avoid repeated processing
                
           else:
                cv2.putText(display, "No ArUco markers detected", (20, 40), display_text_font, display_text_size, red_bgr, display_text_thickness)

           cv2.imshow("Webcam Receiver", display)

           if roi_coordinates is not None: # If there are ROI coordinates:
                
                if not hasattr(receive_message, "roi_padded"): # If "recieve_message" doesn't have the attribute "roi_padded":

                    print("\n[INFO] Calculating padded ROI coordinates...")
                
                    try:
                        roi_padding_px = (aruco_marker_side_length / aruco_marker_size) * aruco_marker_margin # Calculate the padding in pixels

                    except Exception:
                        roi_padding_px = 0

                    start_x, end_x, start_y, end_y = roi_coordinates # Unpack the ROI coordinates

                    # ROI expansion

                    start_x = int(start_x - roi_padding_px)
                    end_x = int(end_x + roi_padding_px)

                    start_y = int(start_y - roi_padding_px)
                    end_y = int(end_y + roi_padding_px)

                    # Minimized ROI coordinates

                    print("\n[INFO] Calculating minimized ROI coordinates...")

                    roi_height = end_y - start_y
                    roi_width = end_x - start_x

                    minimized_roi_height = int(roi_height * minimized_roi_fraction)
                    minimized_roi_width = int(roi_width * minimized_roi_fraction)

                    minimized_start_x = start_x + (minimized_roi_width // 2)
                    minimized_end_x   = minimized_start_x + minimized_roi_width

                    minimized_start_y = start_y + ((roi_height - minimized_roi_height) // 2)
                    minimized_end_y = minimized_start_y + minimized_roi_height

                    print(f"\n[DEBUG] Minimized ROI coordinates: (minimized_start_x = {locals().get('minimized_start_x')}, minimized_end_x = {locals().get('minimized_end_x')}, minimized_start_y = {locals().get('minimized_start_y')}, minimized_end_y = {locals().get('minimized_end_y')})")

                    receive_message.roi_padded = (start_x, end_x, start_y, end_y) # Assigns the attribute "roi_padded" to "recieve_message" with given values

                if start_x < end_x and start_y < end_y: # If the ROI coordinates are valid:

                    cv2.rectangle(display, (start_x, start_y), (end_x, end_y), (green_bgr), roi_rectangle_thickness)
                    cv2.rectangle(display, (minimized_start_x, minimized_start_y), (minimized_end_x, minimized_end_y), (yellow_bgr), minimized_roi_rectangle_thickness)
            
                    roi = frame[start_y:end_y, start_x:end_x] # Extract the ROI from the frame
                    minimized_roi = frame[minimized_start_y:minimized_end_y, minimized_start_x:minimized_end_x] # Extract the minimized ROI from the frame
            
                else: # Else (if they aren't):
                    print("\n[WARNING] Invalid ROI coordinates, creating dummy ROI...")
                    roi = np.zeros((10, 10, 3), dtype = np.uint8) # Create a dummy ROI
                    minimized_roi = roi # Set the minimized ROI to the dummy ROI

                color = dominant_color(minimized_roi) # Get the dominant color in the minimized ROI

                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                print(f"\n[INFO] Dominant color in minimized ROI: {color}")

                if current_state == "aruco_marker_detection" and roi_coordinates is not None and color == "blue":
                    print("[INFO] Starting color calibration...")
                    current_state = "color_calibration"

                cv2.imshow("Webcam Receiver", display)

                # Color calibration

                if current_state == "color_calibration":

                    print(f"\n[INFO] Dominant color in minimized ROI: {color}")

                    try:
                        corrected_ranges = color_offset_calculation(roi)
                        LUT, color_names = build_color_LUT(corrected_ranges)
                        tracker.colors(LUT, color_names)

                    except Exception as e:
                        print("\n[INFO] Color calibration error:", e)

                    current_state = "syncing"
                
                # Syncing

                if current_state == "syncing" and color in ["black", "white"]: # If we're syncing:

                    print("\n[INFO] Trying to sync and get the interval...")

                    try:
                        interval, syncing = sync_interval_detector(color, True) # Try to sync and get the interval
                        print(f"\n[INFO] Interval: {interval} s")

                    except Exception as e:
                        print("\n[WARNING] Sync error:", e)
                        syncing = False
                    
                    if syncing == False:
                        current_state = "decoding"

                # Decoding

                elif current_state == "decoding": # If we're decoding:
                    
                    if not has_printed_decoding_message:
                        print("\n[INFO] Decoding...")
                        has_printed_decoding_message = True

                    recall = False # Initialize recall flag as False
                    end_frame = False # Initialize end_frame flag as False
                    add_frame = False # Initialize add_frame flag as False

                    if last_frame_time is None:
                        last_frame_time = time.time()

                    current_time = time.time()
                    frame_time = current_time - last_frame_time 

                    if interval > 0:
                        if frame_time >= interval:
                            end_frame = True
                            add_frame = True 
                            last_frame_time = current_time 

                    if color in ["white", "black"]: # If the color is white or black:

                        add_frame = True

                    elif color == "red" and last_color != "red": # If the color is red and the last color wasn't red:

                        recall = True # Set recall to True

                    elif color == "red" and last_color == "red":
                        break

                    if recall: # If it's a recall frame:
                        message = decode_bitgrid(roi, frame_bit, add_frame, recall, end_frame) # Decode the bitgrid with recall set to True

                    else: # Else (if it's not a recall frame):
                        decode_bitgrid(roi, frame_bit, add_frame, recall, end_frame)

                    if end_frame:
                        frame_bit += 1 

                last_color = color # Update the last color

                cv2.imshow("ROI", roi)

           if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if current_bit_colors: # If there are colors collected for the current unfinished bit:
            print(f"[INFO] Colors collected for last unfinished bit: {current_bit_colors}")

        if bits: # If there are remaining bits not yet converted:
            print(f"[INFO] Bits not yet converted: {bits}")

        print("\n[INFO] Final message:", message)
    
    finally:
        videoCapture.release()
        cv2.destroyAllWindows() 

# --- Execution ---

if __name__ == "__main__":
    receive_message()
