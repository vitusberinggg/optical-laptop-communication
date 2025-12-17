
# --- Imports ---

# Library modules

import cProfile # Provides deterministic profiling (statistics that describes how often and for how long various parts of a program executes)
import pstats # Module for formatting the profiling into reports

import multiprocessing
import queue

import cv2
import os

import time
import numpy as np

import soundfile
import sounddevice

# Profiling initialization

profiler = cProfile.Profile()
#profiler.enable()

# Non-library modules

from webcam_simulation.cpu_webcam_simulator import VideoProcessCapture
from decoding_pipeline.pipeline import pip_audio
from decoding_pipeline.shared_functions import shared_class

from utilities.color_functions_hcv import (
    color_offset_calculation, tracker, build_color_LUT, dominant_color_hcv, 
    bgr_to_hcv
)

from utilities.color_functions_bgr import dominant_color_bgr
from utilities.screen_alignment_functions import warp_alignment, homography_from_large_markers
from utilities.decoding_functions import sync_interval_detector, decode_bitgrid_hcv_audio
from utilities.audio_functions import audio_reconstructor

from utilities.global_definitions import (
    laptop_webcam_pixel_height, laptop_webcam_pixel_width,
    sender_output_height, sender_output_width,
    roi_window_height, roi_window_width,
    aruco_marker_dictionary, aruco_detector_parameters, 
    aruco_marker_dictionary, aruco_detector_parameters, 
    display_text_font, display_text_size, display_text_thickness,
    green_bgr, red_bgr, yellow_bgr,
    roi_rectangle_thickness, minimized_roi_rectangle_thickness, minimized_roi_fraction, orange_time,
    audio_file
)

# --- Definitions --- 

using_webcam = False

decoded_audio_data = None

# --- Pre-compile numba functions ---

def warmup_numbas():
    # Uses dummies to pre-compile with
    dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_bgr = np.ascontiguousarray(dummy_bgr)
    bgr_to_hcv(dummy_bgr)

# --- Main function ---

def receive_data():

    """
    Receives a message from the sender screen.
    
    Arguments:
        None

    Returns:
        None
    
    """

    # Variable initialization

    bits = ""

    marker_ids = None
    corners = None
    src_pts = None

    wanted_width = 960
    wanted_height = 540

    H = None

    last_color = None
    last_state = None

    last_frame_time = None 
    last_color_time = None
    last_state_time = None

    interval = 0 # Interval between frames in seconds
    frame_waitkey_count = 0

    current_bit_colors = [] # Colors collected for the current bit
    display_warped_roi = None

    has_printed_aruco_detector_message = False
    has_printed_decoding_message = False
    orange_detected = False

    decoded_audio_data = None

    current_state = "aruco_marker_detection"

    # --- Debugging ---

    #"""

    previous_time = time.time()
    frame_count = 0

    #"""

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

    # pre-compiles the numbas
    warmup_numbas()

    try:

        while True:

            read_was_sucessful, frame = videoCapture.read() # Reads a frame from the video capture

            if not read_was_sucessful:

                print("\n[WARNING] Failed to capture a frame, trying again...")
                time.sleep(0.5)
                continue

            # --- Debugging ---

            #"""

            frame_count += 1

            current_time = time.time()

            if current_time - previous_time >= 1.0:
                print(f"[INFO] Loops per second: {frame_count}")
                frame_count = 0
                previous_time = current_time

            #"""

            # --- End of debugging ---

            # --- ArUco marker detection ---

            if current_state == "aruco_marker_detection": # If no ArUco markers have been found:

                try:
                    
                    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Grayscale the frame

                    if not has_printed_aruco_detector_message: # If we haven't already printed the ArUco detector message:
                        print("\n[INFO] Running the ArUco marker detector...")
                        has_printed_aruco_detector_message = True

                    corners, marker_ids, _ = aruco_detector.detectMarkers(grayscaled_frame) # Call the ArUco detector on the grayscaled frame

                    if marker_ids is not None and corners is not None and len(marker_ids) > 0 and H is None: # If markers were detected and there are no H (Homography) yet:
                        H, src_pts = homography_from_large_markers(corners, marker_ids, wanted_width, wanted_height)

                        minimized_roi_height = int(wanted_height * minimized_roi_fraction)
                        minimized_roi_width = int(wanted_width * minimized_roi_fraction)

                        minimized_start_x = minimized_roi_width // 2
                        minimized_end_x   = minimized_start_x + minimized_roi_width

                        minimized_start_y = (wanted_height - minimized_roi_height) // 2
                        minimized_end_y = minimized_start_y + minimized_roi_height

                    
                except Exception:
                    print("\n[WARNING] ArUco detection failed.")

            # --- Display drawings ---
            
            display = frame.copy() # Create a copy of the frame for display purposes

            if marker_ids is not None and len(marker_ids) > 0:

                cv2.aruco.drawDetectedMarkers(display, corners, marker_ids) # Draw the detected markers on the display frame

                cv2.putText(display, f"{len(marker_ids)} ArUco marker(s) detected", (20, 40), display_text_font, display_text_size, green_bgr, display_text_thickness)
                            
            else:
                cv2.putText(display, "No ArUco markers detected", (20, 40), display_text_font, display_text_size, red_bgr, display_text_thickness)

            cv2.imshow("Webcam Receiver", display)

            # --- warped_roi processing ---

            if H is not None: # If there are a H (Homography):
        
                warped_roi = warp_alignment(frame, H, wanted_width, wanted_height)
                minimized_roi = warped_roi[minimized_start_y:minimized_end_y, minimized_start_x:minimized_end_x] # Extract the minimized ROI from the frame

                display_warped_roi = warped_roi.copy()

                cv2.polylines(display, [src_pts.astype(np.int32)], True, (green_bgr), roi_rectangle_thickness)
                cv2.rectangle(display_warped_roi, (minimized_start_x, minimized_start_y), (minimized_end_x, minimized_end_y), (yellow_bgr), minimized_roi_rectangle_thickness)

                warped_roi = np.ascontiguousarray(warped_roi)
                minimized_roi = np.ascontiguousarray(minimized_roi)

                roi_hcv = bgr_to_hcv(warped_roi)
                minimized_roi_hcv = bgr_to_hcv(minimized_roi)
                
                if tracker.LUT is not None:
                    color = dominant_color_hcv(minimized_roi_hcv) # Get the dominant color in the minimized ROI

                else:
                    color = dominant_color_bgr(minimized_roi) # Get the dominant color in the minimized ROI
                            
                # "last_color_time" initialization

                if not hasattr(receive_data, "first_color"): # If "recieve_message" doesn't yet have the attribute "first_color":
                    last_color_time = time.time()
                    receive_data.first_color = ("Get first dominant color")

                # Calculates the time of how long it has been the same color

                if color != last_color and last_color_time is not None: # If the current color isn't the same as the last, and "last_color_time" has a value

                    last_color_time = time.time() - last_color_time

                    print(f"\n[INFO] Dominant color in minimized ROI: {last_color}, lasted for: {last_color_time:.3f}")

                    last_color_time = time.time()
                
                cv2.putText(display, f"Dominant color in minimized ROI: {color}", (20, 100), display_text_font, display_text_size, green_bgr, display_text_thickness) # Puts a text in the GUI of the current dominant color

                cv2.putText(display, f"Current state: {current_state}", (20, 130), display_text_font, display_text_size, red_bgr, display_text_thickness)

                #if current_state == "aruco_marker_detection" and H is not None and color == "blue":
                if current_state == "aruco_marker_detection" and H is not None and color == "red":
                 
                    print("\n[INFO] Starting color calibration...")
                    current_state = "color_calibration"

                cv2.imshow("Webcam Receiver", display)

                # --- Color calibration ---

                if current_state == "color_calibration":
                    
                    if not hasattr(receive_data, "color_calibration"): # If "recieve_message()" doesn't yet have the attribute "color_calibration"

                        try:

                            corrected_ranges = color_offset_calculation(warped_roi)
                            LUT, color_names = build_color_LUT(corrected_ranges)
                            tracker.colors(LUT, color_names)
                            shared_class.push_LUT(LUT, color_names)
                            #shared_class.preallocate_shared_memory(warped_roi)
                            
                            receive_data.color_calibration = ("color calibrated") # Assigns the attribute "color_calibration" to "receive_data()" (to make sure calibration only happens once)

                        except Exception as e:
                            print("\n[INFO] Color calibration error:", e)

                    if hasattr(receive_data, "color_calibration"): # If "recieve_message()" has the attribute "color_calibration":
                        current_state = "syncing"
                
                # --- Syncing ---

                if current_state == "syncing" and color in ["black", "white"]: # If we're syncing:
                    
                    if not hasattr(receive_data, "syncing"):

                        print("\n[INFO] Trying to sync and get the interval...")
                        receive_data.syncing = ("Initialized")

                    try:
                        interval, syncing = sync_interval_detector(color, True) # Try to sync and get the interval

                    except Exception as e:
                        print("\n[WARNING] Sync error:", e)
                        syncing = False
                    
                    if syncing == False:
                        print(f"\n[INFO] Interval: {interval} s")
                        current_state = "end of sync"
                
                # --- End of sync ---

                # --- Blue frame (to prevent early decoding) ---

                elif current_state == "end of sync":
                    if color != "orange" and last_color == "orange":
                        profiler.enable()
                        current_state = "decoding"

                # --- Decoding ---

                elif current_state == "decoding": # If we're decoding:
                    
                    if not has_printed_decoding_message:
                        print("\n[INFO] Decoding...")
                        has_printed_decoding_message = True

                    end_frame = False # Initialize end_frame flag as False
                    add_frame = False # Initialize add_frame flag as False

                    if last_frame_time is None:
                        last_frame_time = time.time()

                    current_time = time.time()
                    frame_time = current_time - last_frame_time 

                    if interval > 0:

                        if frame_time >= interval:
                            end_frame = True
                            last_frame_time = current_time 

                    if color != "orange": # If the color is not orange:

                        add_frame = True

                    elif color == "orange": # If the color is orange and the last color wasn't orange:
                        if not orange_detected:
                            orange_detected = True
                            orange_start_time = time.monotonic()

                        else:
                            if time.monotonic() - orange_start_time > orange_time:
                                profiler.disable()
                                current_state = "waiting for audio"

                    try:
                        frame_data = (roi_hcv, add_frame, end_frame) # Create a tuple with the frame data
                        # Push frames as they arrive
                        shared_class.push_frame(frame_data) # Push the frame to the decoding pipeline

                    except queue.Full: # If the queue is full:
                        pass # Skip
                        
                elif current_state == "waiting for audio":
                    if not hasattr(receive_data, "waiting_for_audio"):
                        print("\n[INFO] Waiting for audio to be decoded...")
                        receive_data.waiting_for_audio = ("Initialized")

                    decoded_audio_data = shared_class.pull_decoded_audio_data()

                    if decoded_audio_data is not None:
                        print("\n[INFO] Audio done")
                        break
                    
                # "last_state_time" initialization

                if not hasattr(receive_data, "first_state"):
                    last_state_time = time.time()
                    receive_data.first_state = ("Get first state")

                # Calculates the time of how long it has been the same state

                if last_state != current_state and last_state_time is not None: # If the current state is different from the last and "last_state_time" has a value:
                    last_state_time = time.time() - last_state_time
                    print(f"\n[INFO] State: {last_state}, lasted for: {last_state_time:.3f}")
                    last_state_time = time.time()

                # --- End of decoding ---

                last_color = color # Update the last color
                last_state = current_state

                if display_warped_roi is not None:
                    if not hasattr(receive_data, "Display_warped_roi"):
                        cv2.namedWindow("Warped roi", cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("Warped roi", roi_window_width, roi_window_height)
                        receive_data.Display_warped_roi = ("Created")
                    cv2.imshow("Warped roi", display_warped_roi)

            # --- End of warped_roi processing ---

            frame_waitkey_count += 1
            if frame_waitkey_count%1 == 0:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if current_bit_colors: # If there are colors collected for the current unfinished bit:
            print(f"[INFO] Colors collected for last unfinished bit: {current_bit_colors}")

        if bits: # If there are remaining bits not yet converted:
            print(f"[INFO] Bits not yet converted: {bits}")

        if decoded_audio_data is not None:

            print(f"\n[INFO] Reconstructing and playing audio...")

            frequency_indices, amplitude_levels = decoded_audio_data

            output_file = audio_reconstructor(frequency_indices, amplitude_levels)

            audio_data, sample_rate = soundfile.read(output_file)
            sounddevice.play(audio_data, sample_rate)
            sounddevice.wait()

        else:
            print("\n[WARNING] No audio data was decoded.")

    finally:
        videoCapture.release()
        cv2.destroyAllWindows() 




# --- Execution ---


if __name__ == "__main__":

    multiprocessing.freeze_support()   # For Windows EXEs, harmless otherwise

    # Video path

    base = os.path.dirname(__file__)
    path = os.path.join(base, "webcam_simulation", "sender_v7_7color.mp4")
    
    # --- Video capture setup ---

    if using_webcam:

        videoCapture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

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
        videoCapture.set(cv2.CAP_PROP_EXPOSURE, -5) # Lower value --> darker
        print(f"\n[INFO] Video capture exposure: {videoCapture.get(cv2.CAP_PROP_EXPOSURE)}")

        # Gain

        videoCapture.set(cv2.CAP_PROP_GAIN, 0) # Disables auto gain

    else:
        videoCapture = VideoProcessCapture(path, False, True, core=[11, 10]) # Initializes a video capture object with a pre-recorded video



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

    # --- ArUco detector setup ---

    aruco_detector = cv2.aruco.ArucoDetector(aruco_marker_dictionary, aruco_detector_parameters)


    # Start pipeline
    pip_audio.start_pipeline(core_decode_worker=[9, 8, 7, 6], core_audio_worker=[5], core_watchdog=[4])

    try:
        receive_data()
    except KeyboardInterrupt:
        print("[Main] Caught Ctrl+C — shutting down pipeline")
        pip_audio.stop_pipeline()

    #profiler.disable()

    stats = pstats.Stats(profiler)
    stats.strip_dirs() # Removes directorys
    stats.sort_stats("cumtime") # Sorts by cumulative time
    stats.print_stats(20) # Prints only top 20 functions
