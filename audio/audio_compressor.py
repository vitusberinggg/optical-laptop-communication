
# --- Imports ---

import numpy as np
import librosa
import soundfile

# --- Definitions ---

target_number_of_frequency_bins = 32
target_number_of_amplitude_levels = 3

target_hop_length = 256
target_sample_rate = 8000 # The audio signal's average number of samples (values) per second

stft_frame_size = 1024

# --- Main function ---

def audio_compressor(input_file):

    """
    
    """

    # Spectrogram creation
    
    y, sample_rate = librosa.load(input_file, sr = target_sample_rate, mono = True)

    spectrogram = librosa.stft(y, n_fft = stft_frame_size, hop_length = hop_length)

    spectrogram_magnitude = np.abs(spectrogram)
    spectrogram_phase = np.angle(spectrogram)

    # Frequency bin reduction

    number_of_frequency_bins = spectrogram_magnitude.shape[0]

    frequency_bin_factor = number_of_frequency_bins // target_number_of_frequency_bins

    reduced_magnitude = spectrogram_magnitude[:frequency_bin_factor * target_number_of_frequency_bins].reshape(target_number_of_frequency_bins, frequency_bin_factor, -1).mean(axis = 1)

    # Amplitude quantization

    normalized_magnitude = reduced_magnitude / reduced_magnitude.max()

    quantized_amplitude_levels = np.linspace(0, 1, target_number_of_amplitude_levels)

    quantized_magnitude = np.digitize(mag_norm, quant_levels) - 1