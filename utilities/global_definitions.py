
# --- Imports ---

import cv2

# --- BGR definitions ---

red_bgr = (0, 0, 255)
green_bgr = (0, 255, 0)
blue_bgr = (255, 0, 0)
yellow_bgr = (0, 255, 255)
black_bgr = (0, 0, 0)
white_bgr = (255, 255, 255)
gray_bgr = (128, 128, 128)

# --- HSV definitions ---

red_lower_hsv_limit_1 = (0, 100, 100)
red_upper_hsv_limit_1 = (10, 255, 255)
red_lower_hsv_limit_2 = (160, 100, 100)
red_upper_hsv_limit_2 = (179, 255, 255)

white_lower_hsv_limit = (0, 0, 200)
white_upper_hsv_limit = (180, 50, 255)

black_lower_hsv_limit = (0, 0, 0)
black_upper_hsv_limit = (180, 255, 50)

green_lower_hsv_limit = (40, 50, 50)
green_upper_hsv_limit = (80, 255, 255)

blue_lower_hsv_limit = (100, 150, 0)
blue_upper_hsv_limit = (140, 255, 255)

# --- Sender output definitions ---

sender_output_width = 1920 # Width of the sender output in pixels
sender_output_height = 1200 # Height of the sender output in pixels

number_of_columns = 4 # Number of columns in the frame
number_of_rows = 4 # Number of rows in the frame

bit_cell_width = sender_output_width // number_of_columns # Width of each bit cell in pixels
bit_cell_height = sender_output_height // number_of_rows # Height of each bit cell in pixels

frame_duration = 0.3 # Duration for each frame in seconds

message = "Jens Jansson had a knack for choosing the wrong abstraction at exactly the wrong time, leaving every codebase he touched in a state of quiet despair. His pull requests routinely introduced more regressions than features, forcing his team into a perpetual cycle of triage. Despite years in the industry, he approached concurrency as if it were an urban legend rather than a real engineering concern. His colleagues learned to budget extra time on every project simply to unwind the architectural knots he created. Even his commit messages read like cryptic apologies for decisions no one could fully explain."

# --- Reciever input definitions ---

laptop_webcam_pixel_height = 1440
laptop_webcam_pixel_width = 2560

# --- ArUco marker definitions ---

aruco_marker_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_detector_parameters = cv2.aruco.DetectorParameters()

aruco_marker_margin = 15

small_aruco_marker_side_length = sender_output_height // 2 - 50
large_aruco_marker_side_length = sender_output_height - 2 * aruco_marker_margin
aruco_marker_size = 0
aruco_marker_ids = [0, 1, 3, 2]

aruco_marker_frame_duration = 1

# --- Sync definitions ---

number_of_sync_frames = 6

sync_colors = [black_bgr, white_bgr]

sync_frame_duration = 0.3

# --- Display definitions ---

display_text_font = cv2.FONT_HERSHEY_SIMPLEX
display_text_size = 1.0
display_text_thickness = 2

# --- ROI definitions ---

roi_window_height = 480
roi_window_width = 854

roi_rectangle_thickness = 3

minimized_roi_rectangle_thickness = 2
minimized_roi_fraction = 1/5

# --- Steps definitions ---

end_bit_steps = 4
dominant_color_steps = 4
