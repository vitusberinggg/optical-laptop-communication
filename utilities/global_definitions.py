
# --- General definitions ---

reference_image_seed = 42 # Seed for the random number generator to generate the reference image
reference_image_duration = 2 # Duration for displaying the reference image in seconds

sender_output_width = 1920 # Width of the sender output in pixels
sender_output_height = 1200 # Height of the sender output in pixels

number_of_columns = 8 # Number of columns in the frame
number_of_rows = 8 # Number of rows in the frame

bit_cell_width = sender_output_width // number_of_columns # Width of each bit cell in pixels
bit_cell_height = sender_output_height // number_of_rows # Height of each bit cell in pixels

frame_duration = 0.3  # Duration for each frame in seconds