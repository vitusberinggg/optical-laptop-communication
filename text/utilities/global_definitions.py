
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
orange_bgr = (0, 140, 255)

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

# --- HCV definitions ---

red_lower_hcv_limit_1 = (0, 100, 100)
red_upper_hcv_limit_1 = (10, 255, 255)
red_lower_hcv_limit_2 = (160, 100, 100)
red_upper_hcv_limit_2 = (179, 255, 255)

white_lower_hcv_limit = (0, 0, 200)
white_upper_hcv_limit = (180, 50, 255)

black_lower_hcv_limit = (0, 0, 0)
black_upper_hcv_limit = (180, 255, 50)

green_lower_hcv_limit = (40, 50, 50)
green_upper_hcv_limit = (80, 255, 255)

blue_lower_hcv_limit = (100, 150, 0)
blue_upper_hcv_limit = (140, 255, 255)

# --- Color maps ---

color_map_2bit = [
    (0, 0, 0),       # 0b00 = Black
    (255, 255, 255), # 0b01 = White
    (255, 0, 0),     # 0b10 = Blue 
    (0, 255, 0),     # 0b11 = Green
]

color_map_3bit = [
    (0, 0, 0),         # 0b000 = Black
    (255, 255, 255),   # 0b001 = White
    (0, 0, 255),       # 0b010 = Red
    (0, 255, 0),       # 0b011 = Green
    (255, 0, 0),       # 0b100 = Blue
    (0, 255, 255),     # 0b101 = Yellow
    (255, 255, 0),     # 0b110 = Cyan
    (255, 0, 255),     # 0b111 = Magenta
]

# --- idx to bits maps ---

idx_to_1bit = {
    1: 0,  # black
    0: 1,  # white
}

idx_to_2bit = {
    1: 0b00,  # black
    0: 0b01,  # white
    5: 0b10,  # blue
    4: 0b11,  # green
}

idx_to_3bit = {
    0: 0b001,  # white
    1: 0b000,  # black
    2: 0b010,  # red1
    3: 0b010,  # red2 (same as red1)
    4: 0b011,  # green
    5: 0b100,  # blue
    6: 0b101,  # yellow
    7: 0b110,  # cyan
    8: 0b111,  # magenta
}

# --- Sender output definitions ---

sender_output_width = 1920 # Width of the sender output in pixels
sender_output_height = 1200 # Height of the sender output in pixels

number_of_columns = 16 # Number of columns in the frame
number_of_rows = 16 # Number of rows in the frame

bit_cell_width = sender_output_width // number_of_columns # Width of each bit cell in pixels
bit_cell_height = sender_output_height // number_of_rows # Height of each bit cell in pixels
bits_per_cell = 3
number_of_colors = bits_per_cell^2 

frame_duration = 0.3 # Duration for each frame in seconds

message = "Hejsan, mamma och pappa! Hare så kult på restaurangen, vad äter ni förresten? Jag uuundrar vad ni äter och vad ska ni äta till efterrätt, det kanske låter GOTT! Men, jag är hemma hos Katrin och Samuel nu? *Suck* Och, och ska titta på Simpsons, OCH... Ha- ...Och ha det så kult på restaurangen, hej då!"

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

number_of_sync_frames = 10

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
minimized_roi_fraction = 1/10
# --- Steps definitions ---

end_bit_steps = 2
dominant_color_steps = 4
