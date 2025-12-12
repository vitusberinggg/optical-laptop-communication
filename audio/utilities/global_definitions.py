
# --- Imports ---

import math
import numpy as np
import cv2

# --- Color definitions ---

red_bgr = (0, 0, 255)
green_bgr = (0, 255, 0)
blue_bgr = (255, 0, 0)
yellow_bgr = (0, 255, 255)
black_bgr = (0, 0, 0)
white_bgr = (255, 255, 255)
gray_bgr = (128, 128, 128)
orange_bgr = (0, 140, 255)
cyan_bgr = (255, 255, 0)
magenta_bgr = (255, 0, 255)

color_map_3_bit = [
    black_bgr,
    white_bgr,
    red_bgr,
    green_bgr,
    blue_bgr,
    yellow_bgr,
    cyan_bgr,
    magenta_bgr
]

sync_colors = [black_bgr, white_bgr]

# --- Sender output definitions ---

audio_file = "audio_files/The Chords - Sh-Boom.mp3"

sender_output_width = 1920 # Width of the sender output in pixels
sender_output_height = 1200 # Height of the sender output in pixels

number_of_columns = 8 # Number of columns in the frame
number_of_rows = 8 # Number of rows in the frame

bit_cell_width = sender_output_width // number_of_columns # Width of each bit cell in pixels
bit_cell_height = sender_output_height // number_of_rows # Height of each bit cell in pixels

number_of_cells = number_of_columns * number_of_rows

bits_per_cell = 3

frame_duration = 0.3 # Duration for each frame in seconds

# --- ArUco marker definitions ---

aruco_marker_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_detector_parameters = cv2.aruco.DetectorParameters()

aruco_marker_margin = 15

large_aruco_marker_side_length = sender_output_height - 2 * aruco_marker_margin

aruco_marker_ids = [0, 1, 3, 2]

aruco_marker_frame_duration = 1

# --- Audio compression definitions ---

number_of_frequencies = 8 # The number of frequencies we reduce the spectrograms to (lower amount --> fewer frequency details --> less data)
number_of_amplitude_levels = 12 # The number of amplitude levels we keep per frequency bin (fewer levels --> coarser dynamics)

quantized_amplitude_levels = np.linspace(0, 1, number_of_amplitude_levels) # Creates the target number of amplitude levels equally spaced between 0 and 1

bits_per_frequency = int(math.log2(number_of_frequencies))
bits_per_amplitude_level = int(math.log2(bits_per_frequency))

bits_per_audio_time_frame = number_of_frequencies * (bits_per_frequency + bits_per_amplitude_level)

bits_per_visual_frame = number_of_cells * bits_per_cell

hop_length = 512 # The amount of samples between each spectrogram frame (shorter hop length --> more overlap between windows --> smoother time reconstruction but more computation)

sample_rate = 8000 # The audio signal's average number of samples (values) per second

seconds_per_time_frame = hop_length / sample_rate # Duration of one STFT hop

frequency_spectrogram_frame_size = 1024 # The amount of samples each spectogram contains (larger windows give higher frequency resolution but lower time resolution)

# --- Sync definitions ---

number_of_sync_frames = 10

sync_frame_duration = 0.3