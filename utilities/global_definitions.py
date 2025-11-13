
# --- General definitions ---

end_frame_color = (0, 0, 255) # Color of the end frame (BGR format)
start_frame_color = (0, 255, 0) # Color of the start frame (BGR format)

# --- Reference image definitions ---

reference_image_seed = 42 # Seed for the random number generator to generate the reference image
reference_image_duration = 2 # Duration for displaying the reference image in seconds
reference_match_threshold = 0.7 # Threshold for reference image matching

# --- Sender output definitions ---

sender_output_width = 1920 # Width of the sender output in pixels
sender_output_height = 1200 # Height of the sender output in pixels

number_of_columns = 8 # Number of columns in the frame
number_of_rows = 8 # Number of rows in the frame

bit_cell_width = sender_output_width // number_of_columns # Width of each bit cell in pixels
bit_cell_height = sender_output_height // number_of_rows # Height of each bit cell in pixels

frame_duration = 0.3  # Duration for each frame in seconds

# --- Reciever input definitions ---

cell_brightness_threshold = 100 # Brightness threshold for determining bit values
end_frame_detection_tolerance = 40 # Tolerance for end frame color detection
start_frame_detection_tolerance = 40 # Tolerance for start frame color detection

samples_per_frame = 3
sample_space = frame_duration / samples_per_frame