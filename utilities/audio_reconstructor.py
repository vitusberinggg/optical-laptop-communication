
# --- Imports ---

# Library imports

import numpy as np
import librosa
import soundfile

# Non-library imports

from utilities.global_definitions import(
    hop_length, sample_rate,
    number_of_frequencies, number_of_amplitude_levels,
    frequency_spectrogram_frame_size,
    quantized_amplitude_levels
)

# --- Main function ---

def audio_reconstructor(frequency_indices_per_time_frame, quantized_amplitude_levels_per_time_frame, spectrogram_phase, output_file = "audio_files/recieved_audio"):

    """
    Reconstructs audio from given frequency and amplitude data.

    Arguments:
        "frequency_indices_per_time_frame" (list of np.arrays): A list of 1D np.arrays containing information about the loudest frequencies for each time frame.  
        "quantized_amplitude_levels_per_time_frame" (list of np.arrays): A list of 1D np.arrays containing information about the quantized amplitude level of each frequency for each time frame.
        "spectrogram_phase" (np.angle): The phase of each frequency bin.
        "output_file" (str): The path of the output file.

    Returns:
        None

    """

    number_of_time_frames = len(frequency_indices_per_time_frame)

    number_of_frequency_bins = frequency_spectrogram_frame_size // 2 + 1 # Calculates the number of frequency bins using the 1 + n_fft / 2 formula

    reconstructed_spectrogram = np.zeros((number_of_frequency_bins, number_of_time_frames)) # Initialized as empty

    for time_frame in range(number_of_time_frames): # For each time frame:

        frequency_indices = frequency_indices_per_time_frame[time_frame] # Extract the frequency indices
        amplitude_indices = quantized_amplitude_levels_per_time_frame[time_frame] # Extract the amplitude indices
        
        amplitude_values = quantized_amplitude_levels[amplitude_indices] # Map amplitude indices back to actual values
        
        reconstructed_spectrogram[frequency_indices, time_frame] = amplitude_values # Places the amplitudes at their corresponding frequency bins

    reconstructed_spectrogram = reconstructed_spectrogram * np.exp(1j * spectrogram_phase)

    reconstructed_audio_signal = librosa.istft(reconstructed_spectrogram, hop_length = hop_length)

    soundfile.write(output_file, reconstructed_audio_signal, sample_rate)

    return