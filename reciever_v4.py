
# --- Imports ---

import cProfile
import cv2
import time
import numpy as np

from recievers.webCamSim import VideoThreadedCapture
from utilities.color_functions import dominant_color, tracker
from utilities import detection_functions, screen_alignment_functions, decoding_functions
from utilities.global_definitions import (
    sender_output_height, sender_output_width,
    roi_window_height, roi_window_width,
    laptop_webcam_pixel_height, laptop_webcam_pixel_width,
    aruco_marker_dictionary, aruco_detector_parameters, aruco_marker_size, aruco_marker_margin,
    display_text_font, display_text_size, display_text_thickness,
    green_bgr, red_bgr
)

# --- Video capture setup ---

videoCapture = VideoThreadedCapture(r"C:\Users\ejadmax\code\optical-laptop-communication\recievers\intervals_test.mp4") # For video test
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
    interval = 0

    decoding = False
    current_bit_colors = []
    roi_coordinates = None
    frame_bit = 0

    previous_time = time.time()
    frame_count = 0

#   ArUco marker detection

    print("Receiver started — searching for ArUco markers...")

    while True:

        read_was_sucessful, frame = videoCapture.read()

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

        if arucos_found is False:

            grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            corners, marker_ids, _ = aruco_detector.detectMarkers(grayscaled_frame)

            if marker_ids is not None and len(marker_ids) > 0 and roi_coordinates is None:
                roi_coordinates, aruco_marker_width, aruco_marker_height = screen_alignment_functions.roi_alignment(frame)

#       Display drawings
        
        display = frame.copy()

        if marker_ids is not None and len(marker_ids) > 0:

            cv2.aruco.drawDetectedMarkers(display, corners, marker_ids)

            cv2.putText(display, f"{len(marker_ids)} ArUco marker(s) detected", (20, 40), display_text_font, display_text_size, green_bgr, display_text_thickness)
            
            arucos_found = True
            marker_ids = None
            
        else:
            cv2.putText(display, "No ArUco markers detected", (20, 40), display_text_font, display_text_size, red_bgr, display_text_thickness)

        if roi_coordinates is not None and not hasattr(receive_message, "roi_padded"): # If there are ROI coordinates and "recieve_message" doesn't have the attribute "roi_padded":
            
            x0, x1, y0, y1 = roi_coordinates
            x0 = int(x0 - (aruco_marker_width / aruco_marker_size) * aruco_marker_margin)
            y0 = int(y0 - (aruco_marker_height / aruco_marker_size) * aruco_marker_margin)
            x1 = int(x1 + (aruco_marker_width / aruco_marker_size) * aruco_marker_margin)
            y1 = int(y1 + (aruco_marker_height / aruco_marker_size) * aruco_marker_margin)

            dX = (x1 - x0)/2
            dY = (y1 - y0)/2

            sx0 = int(x0 - dX)
            sy0 = int(y0 - dY)
            sx1 = int(x1 + dX)
            sy1 = int(y1 + dY)

            receive_message.roi_padded = (x0, x1, y0, y1)

            if x0 < x1 and y0 < y1:
                cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 255), 2)

        roi = frame[y0:y1, x0:x1] if roi_coordinates is not None else np.zeros((10, 10, 3), dtype=np.uint8)
        small_roi = frame[sy0:sy1, sx0:sx1] if roi_coordinates is not None else np.zeros((10, 10, 3), dtype=np.uint8)

        color = dominant_color(small_roi)

        cv2.imshow("Webcam Receiver", display)
        cv2.imshow("ROI", roi)

        # --- Waiting for green ---

        if waiting_for_sync:

            if color == "green" and last_color != "green":
                print("Green detected — waiting for sync...")
                tracker.reset()

            elif color != "green" and last_color == "green":
                print("Green ended — starting sync!")
                tracker.reset()
                waiting_for_sync = False
                syncing = True
                decoding = True

        # --- Sync ---

        elif syncing:

            interval, syncing = decoding_functions.sync_receiver(small_roi, True)

            

        # --- Decode ---

        elif decoding:

            recall = False
            end_frame = False
            add_frame = False

            if color == "blue" and last_color != "blue":
                
                end_frame = True
                add_frame = True


            elif color in ["white", "black"]:
                
                # add_frame → add frame to array

                add_frame = True

            elif color == "red" and last_color != "red":

                recall = True    

            if recall:
                message = decoding_functions.decode_bitgrid(roi, frame_bit, add_frame, recall, end_frame)
            else:
                decoding_functions.decode_bitgrid(roi, frame_bit, add_frame, recall, end_frame)

            if end_frame:
                frame_bit += 1

        last_color = color
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    if current_bit_colors:
        print(f"Colors collected for last unfinished bit: {current_bit_colors}")

    if bits:
        print(f"Remaining bits not yet converted: {bits}")

    print("Final message:", message)
    print(f"Interval: {interval}s")
    videoCapture.release()
    cv2.destroyAllWindows()

# --- Run ---
cProfile.run("receive_message()")
