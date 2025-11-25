
# --- Imports ---

import cv2

# --- General definitions ---

red_bgr = (0, 0, 255)
green_bgr = (0, 255, 0)
blue_bgr = (255, 0, 0)

black = (0, 0, 0)
white = (255, 255, 255)

mask_frame_hsv_lower_limit = [40, 100, 100] # [Hue, Saturation, Value]
mask_frame_hsv_upper_limit = [80, 255, 255] # [Hue, Saturation, Value]

# --- ECC allignment definitions ---

reference_image_seed = 42 # Seed for the random number generator to generate the reference image
reference_image_duration = 1 # Duration for displaying the reference image in seconds
reference_match_threshold = 0.7 # Threshold for reference image matching

ecc_allignment_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)

# --- Sender output definitions ---

sender_output_width = 1920 # Width of the sender output in pixels
sender_output_height = 1200 # Height of the sender output in pixels

sender_screen_size_threshold = 0.1 # How much of the frame the sender screen needs to cover in order to trigger the mask creation

number_of_columns = 8 # Number of columns in the frame
number_of_rows = 8 # Number of rows in the frame

bit_cell_width = sender_output_width // number_of_columns # Width of each bit cell in pixels
bit_cell_height = sender_output_height // number_of_rows # Height of each bit cell in pixels

frame_duration = 0.3  # Duration for each frame in seconds

# --- Reciever input definitions ---

laptop_webcam_pixel_height = 1440
laptop_webcam_pixel_width = 2560
total_pixel_count = laptop_webcam_pixel_height * laptop_webcam_pixel_width

roi_window_height = 200
roi_window_width = 200

cell_brightness_threshold = 100 # Brightness threshold for determining bit values
end_frame_detection_tolerance = 40 # Tolerance for end frame color detection
start_frame_detection_tolerance = 40 # Tolerance for start frame color detection

samples_per_frame = 3
sample_space = frame_duration / samples_per_frame

# --- ArUco marker definitions ---

aruco_marker_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_detector_parameters = cv2.aruco.DetectorParameters()

aruco_marker_size = min(sender_output_width, sender_output_height) // 2 - 50
aruco_marker_margin = 15
aruco_marker_ids = [0, 1, 3, 2]

aruco_marker_frame_duration = 1

# --- Sync definitions ---

number_of_sync_frames = 6

sync_colors = [black, white]

sync_frame_duration = 0.2

# --- Display text definitions ---

display_text_font = cv2.FONT_HERSHEY_SIMPLEX
display_text_size = 1.0
display_text_thickness = 2