
# --- Imports ---

import cProfile
import pstats

# Enables profiling if run as main module
if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()

import threading
import queue
import cv2
import time
import numpy as np

from webcam_simulation.webcamSimulator import VideoThreadedCapture

from utilities.color_functions_v3_1 import color_offset_calculation, tracker, build_color_LUT, dominant_color_hsv, dominant_color_bgr
from utilities.screen_alignment_functions import roi_alignment_for_large_markers
from utilities.decoding_functions_v3_1 import sync_interval_detector, decode_bitgrid
from utilities.accuracy_calculator import accuracy_calculator
from utilities.global_definitions import (
    laptop_webcam_pixel_height, laptop_webcam_pixel_width,
    sender_output_height, sender_output_width,
    roi_window_height, roi_window_width,
    aruco_marker_dictionary, aruco_detector_parameters, large_aruco_marker_side_length, aruco_marker_margin,
    aruco_marker_dictionary, aruco_detector_parameters, large_aruco_marker_side_length, aruco_marker_margin,
    display_text_font, display_text_size, display_text_thickness,
    green_bgr, red_bgr, yellow_bgr,
    roi_rectangle_thickness, minimized_roi_rectangle_thickness, minimized_roi_fraction
)

# Defenitions 

using_webcam = False
debug_bytes = False
 
# --- Video capture setup ---

if using_webcam:

    videoCapture = cv2.VideoCapture(0, cv2.CAP_DSHOW) # For live webcam

    # Resolution

    videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, laptop_webcam_pixel_width)
    videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, laptop_webcam_pixel_height)

    # White balance
    """
    videoCapture.set(cv2.CAP_PROP_AUTO_WB, 0) # Disables auto white balance
    videoCapture.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 3000)
    print(f"\n[INFO] Video capture white balance: {videoCapture.get(cv2.CAP_PROP_WB_TEMPERATURE)}")
    """
    # Exposure

    videoCapture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # Disables auto exposure
    videoCapture.set(cv2.CAP_PROP_EXPOSURE, -5) # Lower value = Darker
    print(f"\n[INFO] Video capture exposure: {videoCapture.get(cv2.CAP_PROP_EXPOSURE)}")

    # Gain

    videoCapture.set(cv2.CAP_PROP_GAIN, 0) # Disables auto gain

else:
    videoCapture = VideoThreadedCapture(r"C:\Users\ejadmax\code\optical-laptop-communication\webcam_simulation\sender_v5.mp4") # For video test

if not videoCapture.isOpened():
    print("\n[WARNING] Couldn't start video capture.")
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

# --- Pre-compile functions ---

def warmup_all():
    from utilities.color_functions_v3_1 import bitgrid_majority_calc

    dummy_merged = np.zeros((2, 2, 8, 16, 10), dtype=np.uint8)
    bitgrid_majority_calc(dummy_merged, 5)

# --- Threading setup ---

frame_queue = queue.Queue(maxsize=100)
last_queue_debug = 0
decode_last_time = time.time()
decoded_message = None
stop_thread = False

debug_worker = False
debug_watchdog = False

# --- Decoding worker thread ---

def decoding_worker():
    global decoded_message, last_queue_debug, decode_last_time
    while not stop_thread or not frame_queue.empty():

        # [Failsafe] Skip if no frames are available
        try:
            hsv_roi, recall, add_frame, end_frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        if debug_worker:
            # [DEBUG] Print queue size every 0.5 seconds
            now = time.time()
            if now - last_queue_debug > 0.5:
                print(f"[DEBUG] Decode thread queue size = {frame_queue.qsize()}")
                last_queue_debug = now
            t0 = time.time()

        # Decode bitgrid
        if recall:
            result = decode_bitgrid(
                hsv_roi, add_frame, recall, end_frame, debug_bytes
            )

            # Only accept valid non-empty results
            if isinstance(result, str) and result.strip() != "":
                decoded_message = result

        else:
            decode_bitgrid(
                hsv_roi, add_frame, recall, end_frame, debug_bytes
            )

        if debug_watchdog:
            # [DEBUG] Helps watchdog to see if the decode works or not
            decode_last_time = time.time()

        if debug_worker:
            # [DEBUG] Print decode time every 0.5 seconds
            t1 = time.time()
            if t1 - last_queue_debug > 0.5:
                print(f"[DEBUG] Decode time: {(t1 - t0)*1000:.2f} ms")

# --- Start decoding thread ---

decode_thread = threading.Thread(target=decoding_worker, daemon=True)
decode_thread.start()

# --- Watchdog setup ---

watchdog_on = False

# --- Watchdog function ---

def watchdog():
    while watchdog_on:
        # [WARNING] Check if decode thread is stalled or starved
        if time.time() - decode_last_time > 1.0:
            print("[WARNING] Decode thread is stalled or starving (no frames processed)!")
        time.sleep(0.2)

# --- Start watchdog thread ---

watch_thread = threading.Thread(target=watchdog, daemon=True)
watch_thread.start()

# --- Main function ---

def receive_message():

    """
    Receives a message from the sender screen.
    
    Arguments:
        None

    Returns:
        None
    
    """

    # Global variables

    global watchdog_on

    # Variable initialization

    bits = ""

    marker_ids = None
    corners = None

    last_color = None

    last_frame_time = None 
    last_color_time = None

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

    """

    # --- End of debugging ---
    
    print("\n[INFO] Receiver started")

    if using_webcam:
        actual_capture_width = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_capture_height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)

        print(f"\n[INFO] Video capture resolution: {round(actual_capture_width)} x {round(actual_capture_height)}")
        
    # --- Debugging ---

    """

    print(f"[DEBUGGING] ArUco marker dictionary: {type(aruco_marker_dictionary)}")
    print(f"[DEBUGGING] ArUco detector parameters: {type(aruco_detector_parameters)}")   

    """

    # --- End of debugging ---

    try:

        while True:

            read_was_sucessful, frame = videoCapture.read() # Reads a frame from the video capture

            if not read_was_sucessful:

                print("\n[WARNING] Failed to capture a frame, trying again...")
                time.sleep(0.5)
                continue

            # --- Debugging ---

            """

            frame_count += 1

            current_time = time.time()

            if current_time - previous_time >= 1.0:
                print(f"[INFO] Loops per second: {frame_count}")
                frame_count = 0
                previous_time = current_time

            """

            # --- End of debugging ---

            # --- ArUco marker detection ---

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

            # --- Display drawings ---
            
            display = frame.copy() # Create a copy of the frame for display purposes

            if marker_ids is not None and len(marker_ids) > 0:

                cv2.aruco.drawDetectedMarkers(display, corners, marker_ids) # Draw the detected markers on the display frame

                cv2.putText(display, f"{len(marker_ids)} ArUco marker(s) detected", (20, 40), display_text_font, display_text_size, green_bgr, display_text_thickness)
            
                # marker_ids = None # Reset marker IDs to avoid repeated processing
                
            else:
                cv2.putText(display, "No ArUco markers detected", (20, 40), display_text_font, display_text_size, red_bgr, display_text_thickness)

            cv2.imshow("Webcam Receiver", display)

            # --- ROI processing ---

            if roi_coordinates is not None: # If there are ROI coordinates:
                
                if not hasattr(receive_message, "roi_padded"): # If "recieve_message" doesn't have the attribute "roi_padded":

                    print("\n[INFO] Calculating padded ROI coordinates...")
                
                    try:
                        roi_padding_px = (aruco_marker_side_length / large_aruco_marker_side_length) * aruco_marker_margin # Calculate the padding in pixels
                    except Exception:
                        roi_padding_px = 0

                        

                    start_x, end_x, start_y, end_y = roi_coordinates # Unpack the ROI coordinates

                    # ROI expansion

                    print("\n[INFO] Calculating ROI coordinates...")

                    start_x = int(start_x - roi_padding_px)
                    end_x = int(end_x + roi_padding_px)

                    start_y = int(start_y - roi_padding_px)
                    end_y = int(end_y + roi_padding_px)

                    print(f"[INFO] ROI coordinates: (start_x = {locals().get('start_x')}, end_x = {locals().get('end_x')}, start_y = {locals().get('start_y')}, end_y = {locals().get('end_y')})")

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

                    print(f"[INFO] Minimized ROI coordinates: (minimized_start_x = {locals().get('minimized_start_x')}, minimized_end_x = {locals().get('minimized_end_x')}, minimized_start_y = {locals().get('minimized_start_y')}, minimized_end_y = {locals().get('minimized_end_y')})")

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

                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                minimized_roi_hsv = cv2.cvtColor(minimized_roi, cv2.COLOR_BGR2HSV)

                if tracker.LUT is not None:
                    color = dominant_color_hsv(minimized_roi_hsv) # Get the dominant color in the minimized ROI
                else:
                    color = dominant_color_bgr(minimized_roi) # Get the dominant color in the minimized ROI
                
                if not hasattr(receive_message, "first_color"):
                    last_color_time = time.time()
                    receive_message.first_color = ("Get first dominant color")

                if last_color != color and last_color_time is not None:
                    last_color_time = time.time() - last_color_time
                    print(f"\n[INFO] Dominant color in minimized ROI: {last_color}, lasted for: {last_color_time:.3f}")
                    last_color_time = time.time()

                if current_state == "aruco_marker_detection" and roi_coordinates is not None and color == "blue":
                    print("\n[INFO] Starting color calibration...")
                    current_state = "color_calibration"

                cv2.imshow("Webcam Receiver", display)

                # --- Color calibration ---

                if current_state == "color_calibration":
                    
                    # Checks if receive_message() has an attribute that is called "color_calibration"
                    if not hasattr(receive_message, "color_calibration"):
                        try:
                            corrected_ranges = color_offset_calculation(roi)
                            LUT, color_names = build_color_LUT(corrected_ranges)
                            tracker.colors(LUT, color_names)

                            # Warming up numba for use
                            warmup_all()
                            
                            # Gives receive_message() an attribute so that it calibrates colors only once
                            receive_message.color_calibration = ("color calibrated")

                        except Exception as e:
                            print("\n[INFO] Color calibration error:", e)

                    if hasattr(receive_message, "color_calibration"):
                        current_state = "syncing"
                
                # --- Syncing ---

                if current_state == "syncing" and color in ["black", "white"]: # If we're syncing:
                    
                    if not hasattr(receive_message, "syncing"):
                        print("\n[INFO] Trying to sync and get the interval...")
                        receive_message.syncing = ("Initialized")

                        if debug_watchdog:
                            print("\n[DEBUG] Watchdog on\n")
                            watchdog_on = True

                    try:
                        interval, syncing = sync_interval_detector(color, True) # Try to sync and get the interval

                    except Exception as e:
                        print("\n[WARNING] Sync error:", e)
                        syncing = False
                    
                    if syncing == False:
                        print(f"\n[INFO] Interval: {interval} s")
                        current_state = "end of sync"
                
                # --- End of sync ---

                elif current_state == "end of sync":
                    # A frame between sync and decoding so that it doesn't decode during a sync frame
                    if color != "blue" and last_color == "blue":
                        current_state = "decoding"

                # --- Decoding ---

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
                        
                        print("\n[INFO] Red detected — waiting for decode thread to process all frames...")
                        while not frame_queue.empty(): # Waits for the frame queue to be empty
                            time.sleep(0.005)

                        recall = True # Set recall to True
                        print("\n[INFO] Frames finished — recalling message...")

                    try:
                        frame_queue.put_nowait((roi_hsv.copy(), recall, add_frame, end_frame))
                    except queue.Full:
                        pass  # [Failsafe] Skip if queue is full

                    while recall and (decoded_message is None or decoded_message.strip() == ""):
                        time.sleep(0.05)

                    if decoded_message is not None:
                        print("\n[INFO] Decoding finished.")
                        break

                # --- End of decoding ---

                last_color = color # Update the last color

                cv2.imshow("ROI", roi)

            # --- End of ROI processing ---

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if current_bit_colors: # If there are colors collected for the current unfinished bit:
            print(f"[INFO] Colors collected for last unfinished bit: {current_bit_colors}")

        if bits: # If there are remaining bits not yet converted:
            print(f"[INFO] Bits not yet converted: {bits}")

        print(f"\n[INFO] Final message: {decoded_message}")

        accuracy_percentage = accuracy_calculator(decoded_message)

        print(f"\n[INFO] Accuracy: {accuracy_percentage} %")

    finally:
        videoCapture.release()
        cv2.destroyAllWindows() 

# --- Execution ---

if __name__ == "__main__":
    receive_message()

    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.strip_dirs()        # remove full paths
    stats.sort_stats("cumtime")  # sort by cumulative time
    stats.print_stats(20)      # print only top 20 functions
