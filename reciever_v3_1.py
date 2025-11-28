
# --- Imports ---

import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

import threading
import queue

import cv2
import time
import numpy as np

from webcam_simulation.webcamSimulator import VideoThreadedCapture, VideoCaptureSingle
from utilities.color_functions_v3_1 import dominant_color, tracker, build_color_LUT, bitgrid_majority_calc, dominant_color_numba
from utilities import decoding_functions_v3_1, screen_alignment_functions
from utilities.global_definitions import (
    sender_output_height, sender_output_width,
    laptop_webcam_pixel_height, laptop_webcam_pixel_width,
    aruco_marker_dictionary, aruco_marker_size, aruco_marker_margin
)

# function that pre-compiles the numba functions to prevent lag on initial launch with them
def warmup_all():
    from utilities.color_functions_v3_1 import bitgrid_majority_calc, dominant_color_numba

    # Warm up bitgrid majority calc
    dummy_merged = np.zeros((2, 8, 16, 10), dtype=np.uint8)
    bitgrid_majority_calc(dummy_merged, 5)

    # Warm up dominant color analyzer
    dummy_classes = np.zeros((100,), dtype=np.uint8)
    dominant_color_numba(dummy_classes, 5)


# --- ArUco setup (match sender) ---
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


frame_queue = queue.Queue(maxsize=100)
last_queue_debug = 0
decode_last_time = time.time()
decoded_message = ""
stop_thread = False

def decoding_worker():
    global decoded_message, last_queue_debug, decode_last_time
    while not stop_thread or not frame_queue.empty():
        try:
            hsv_roi, recall, add_frame, end_frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        now = time.time()
        if now - last_queue_debug > 0.5:
            print(f"[DEBUG] Decode thread queue size = {frame_queue.qsize()}")
            last_queue_debug = now
        
        t0 = time.time()
        if recall:
            decoded_message = decoding_functions_v3_1.decode_bitgrid(
                hsv_roi, add_frame, recall, end_frame
            )
        else:
            decoding_functions_v3_1.decode_bitgrid(
                hsv_roi, add_frame, recall, end_frame
            )

        decode_last_time = time.time()  # helps watchdog to see if decode works or not


        t1 = time.time()

        # Print timing occasionally
        if t1 - last_queue_debug > 0.5:
            print(f"[DEBUG] Decode time: {(t1 - t0)*1000:.2f} ms")

# Start decoding thread
decode_thread = threading.Thread(target=decoding_worker, daemon=True)
decode_thread.start()

watchdog_on = False

def watchdog():
    while watchdog_on:
        if time.time() - decode_last_time > 1.0:
            print("[WARNING] Decode thread is stalled or starving (no frames processed)!")
        time.sleep(0.2)

watch_thread = threading.Thread(target=watchdog, daemon=True)
watch_thread.start()


# --- Main function ---

def receive_message():

    """
    Receives a message from the sender screen via webcam/video.
    
    Arguments:
        None

    Returns:
        None
    
    """

    global watchdog_on

    message = ""
    arucos_found = False
    keep_looking = False

    last_color = None
    color = None

    aligning_screen = True
    syncing = False
    color_calibration = False
    interval = 0
    time_since_red = 0

    decoding = False
    roi_coords = None

    prev_time = time.time()
    frame_count = 0

    x0 = 0
    y0 = 0
    x1 = 0
    y1 = 0

    # --- To help pre-compile the numba functions ---
    # also prevents exceptions caused by LUT or color_names not having any values
    
    
    corrected_ranges = {
            "red":    (np.array([0, 100, 100]), np.array([10, 255, 255])),  # hue 0–10
            "red2":   (np.array([160, 100, 100]), np.array([179, 255, 255])),  # hue 160–179
            "white":  (np.array([0, 0, 220]), np.array([180, 25, 255])),
            "black":  (np.array([0, 0, 0]), np.array([180, 255, 35])),
            "green":  (np.array([45, 80, 80]), np.array([75, 255, 255])),
            "blue":   (np.array([95, 120, 70]), np.array([130, 255, 255]))

            
        }
    

    LUT, color_names = build_color_LUT(corrected_ranges)
    tracker.colors(LUT, color_names)

    warmup_all()

    # --- Setup capture ---
    cap = VideoThreadedCapture(r"C:\Users\ejadmax\code\optical-laptop-communication\webcam_simulation\sender_v3_video.mp4")
    # For live webcam test instead of video, use:
    #cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("Error: Could not open camera/video.")
        exit()

    cv2.namedWindow("Webcam Receiver", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Webcam Receiver", 1920, 1200)

    cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ROI", 192, 120)

    # Grab one initial frame so cap is "warmed up"
    while True:

        ret, frame = cap.read()

        if ret:
            break
        time.sleep(0.01)

    print("Receiver started — waiting for Arucos...")

    while True:

        ret, frame = cap.read()

        if not ret:

            print("Error: Failed to capture frame.")
            continue

        frame_count += 1
        current_time = time.time()
        if current_time - prev_time >= 1.0:
            print(f"Loops per second: {frame_count}")
            frame_count = 0
            prev_time = current_time

        # ---------- ArUco detection on the frame ----------

        if arucos_found is False:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = aruco_detector.detectMarkers(gray)

            if ids is not None and len(ids) > 0 and roi_coords is None:
                roi_coords, marker_w, marker_h = screen_alignment_functions.roi_alignment(frame)

        elif keep_looking:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = aruco_detector.detectMarkers(gray)

        #rejected_count = 0 if rejected is None else len(rejected)
        #print("ids:", None if ids is None else ids.flatten(), "rejected:", rejected_count)



        # ---- DRAW ARUCO INFO ON THE SAME FRAME ----

        display = frame.copy()

        if ids is not None and len(ids) > 0 and not arucos_found:

            cv2.aruco.drawDetectedMarkers(display, corners, ids)

            cv2.putText(display,
                        f"{len(ids)} ArUco marker(s) detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2)
            arucos_found = True
            
        else:
            cv2.putText(display,
                        "No ArUco markers detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2)

        if roi_coords is not None and not hasattr(receive_message, "roi_padded"):
            
            x0, x1, y0, y1 = roi_coords
            x0 = int(x0 - (marker_w/aruco_marker_size) * aruco_marker_margin)
            y0 = int(y0 - (marker_h/aruco_marker_size) * aruco_marker_margin)
            x1 = int(x1 + (marker_w/aruco_marker_size) * aruco_marker_margin)
            y1 = int(y1 + (marker_h/aruco_marker_size) * aruco_marker_margin)

            dX = ((x1 - x0) * 2)/5
            dY = ((y1 - y0) * 2)/5

            if ((x1 - x0)/5 * (y1 - y0)/5) < 16:
                # Padded square ROI
                sx0 = int(x0 + dX)
                sy0 = int(y0 + dY)
                sx1 = int(x1 - dX)
                sy1 = int(y1 - dY)

            else:
                # Centered square ROI
                sx0 = int(x1/2 - 2)
                sy0 = int(y1/2 - 2)
                sx1 = int(x1/2 + 2)
                sy1 = int(y1/2 + 2)
            receive_message.roi_padded = (x0, x1, y0, y1)

        if x0 < x1 and y0 < y1:
            cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 255), 2)
            cv2.rectangle(display, (sx0, sy0), (sx1, sy1), (255, 0, 255), 2)

        cv2.imshow("Webcam Receiver", display)

        if not aligning_screen:
            roi = frame[y0:y1, x0:x1] if roi_coords is not None else np.zeros((10, 10, 3), dtype=np.uint8)

            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            small_roi = hsv_roi[sy0:sy1, sx0:sx1] if roi_coords is not None else np.zeros((10, 10, 3), dtype=np.uint8)

            cv2.imshow("ROI", roi)

        # --- Waiting for arucos ---

        if aligning_screen:

            if arucos_found and ids is not None and len(ids) > 0:
                if not hasattr(receive_message, "walla"):
                    print("Arucos detected — waiting for color calibration...")
                    receive_message.walla = ("bingbing")
                tracker.reset()
                keep_looking = True

            elif keep_looking and ids is None:
                print("Arucos disappeared — starting color calibration!")
                tracker.reset()
                aligning_screen = False
                keep_looking = False
                color_calibration = True

        elif color_calibration:

            #LUT, color_names = build_color_LUT(corrected_ranges)
            #tracker.colors(LUT, color_names)

            color = dominant_color(small_roi)

            if color != "green":
                color_calibration = False
                syncing = True

        # --- Sync ---

        if syncing:
            color = dominant_color(small_roi)
            interval, syncing = decoding_functions_v3_1.sync_interval_detector(color, True)
            if not syncing:
                decoding = True
                watchdog_on = True
                print("\n[DEBUG] Watchdog initialized\n")

        # --- Decode ---

        elif decoding:

            recall = False
            end_frame = False
            add_frame = False
            
            color = dominant_color(small_roi)

            if color == "blue" and last_color != "blue":
                
                end_frame = True
                add_frame = True

            elif color in ["white", "black"]:
                
                # add_frame → add frame to array

                add_frame = True

            elif color == "red" and last_color != "red":

                recall = True
                time_since_red = time.time()  

            try:
                frame_queue.put_nowait((hsv_roi.copy(), recall, add_frame, end_frame))
            except queue.Full:
                pass  # skip if queue is full

        if color is not None:
            last_color = color

        if (time.time() - time_since_red > 1.0) and time_since_red != 0:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    print("Final message:", decoded_message)
    print(f"Interval: {interval}s")
    cap.release()
    cv2.destroyAllWindows()

# --- Run ---
receive_message()

profiler.disable()

stats = pstats.Stats(profiler)
stats.strip_dirs()        # remove full paths
stats.sort_stats("cumtime")  # sort by cumulative time
stats.print_stats(20)      # print only top 20 functions
