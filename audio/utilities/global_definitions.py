
# --- Imports ---

import math

# --- Sender output definitions ---

number_of_columns = 8 # Number of columns in the frame
number_of_rows = 8 # Number of rows in the frame

number_of_cells = number_of_columns * number_of_rows

bits_per_cell = 3

# --- Audio compression definitions ---

number_of_frequencies = 8 # The number of frequencies we reduce the spectrograms to (lower amount --> fewer frequency details --> less data)
number_of_amplitude_levels = 12 # The number of amplitude levels we keep per frequency bin (fewer levels --> coarser dynamics)

bits_per_frequency = math.log2(number_of_frequencies)
bits_per_amplitude_level = math.log2(bits_per_frequency)

bits_per_audio_time_frame = number_of_frequencies * (bits_per_frequency + bits_per_amplitude_level)

bits_per_visual_frame = number_of_cells * bits_per_cell

hop_length = 512 # The amount of samples between each spectrogram frame (shorter hop length --> more overlap between windows --> smoother time reconstruction but more computation)

sample_rate = 8000 # The audio signal's average number of samples (values) per second

seconds_per_time_frame = hop_length / sample_rate # Duration of one STFT hop

frequency_spectrogram_frame_size = 1024 # The amount of samples each spectogram contains (larger windows give higher frequency resolution but lower time resolution)