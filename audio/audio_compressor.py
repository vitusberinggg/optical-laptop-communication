
# --- Imports ---

# Library imports

import numpy as np # Library for handling numerical arrays, needed to manipulate matrices
import librosa # High-level audio processing library to load audio files, compute frequency spectrograms, resample etc.

# --- Definitions ---

target_number_of_frequency_bins = 32 # The number of frequency bands we reduce the spectrograms to (lower amount --> fewer frequency details --> less data)
target_number_of_amplitude_levels = 3 # The number of amplitude levels we keep per frequency bin (fewer levels --> coarser dynamics)

hop_length = 256 # The amount of samples between each spectrogram frame (shorter hop length --> more overlap between windows --> smoother time reconstruction but more computation)

target_sample_rate = 8000 # The audio signal's average number of samples (values) per second

frequency_spectrogram_frame_size = 1024 # The amount of samples each spectogram contains (larger windows give higher frequency resolution but lower time resolution)

input_file = "The Chords - Sh-Boom.mp3"

# --- Main function ---

def audio_compressor(input_file):

    """
    
    """

    # Spectrogram creation
    
    y, sample_rate = librosa.load(input_file, sr = target_sample_rate, mono = True) # Reads the audio file

    spectrogram = librosa.stft(y, n_fft = frequency_spectrogram_frame_size, hop_length = hop_length) # Computes the STFT, which converts the time-domain signal into a complex matrix where each row corresponds to a frequency bin, and each column corresponds to a time frame

    spectrogram_magnitude = np.abs(spectrogram) # The absolute value (amplitude) of each frequency bin at each time frame
    spectrogram_phase = np.angle(spectrogram) # The phase of each frequency bin (needed for exact waveform reconstruction, as quantizing causes artifacts)

    # Frequency bin reduction

    number_of_frequency_bins = spectrogram_magnitude.shape[0] # Gets the original number of frequency bins

    frequency_bin_factor = number_of_frequency_bins // target_number_of_frequency_bins # Calculates the amount of frequency bins needed to be grouped together

    reduced_magnitude = spectrogram_magnitude[:frequency_bin_factor * target_number_of_frequency_bins].reshape(target_number_of_frequency_bins, frequency_bin_factor, -1).mean(axis = 1)

    normalized_magnitude = reduced_magnitude / reduced_magnitude.max() # Normalizes the magnitude by dividing it by its maximum value to ensure that quantization levels are interpreted relative to the maximum energy present

    # Amplitude quantization

    quantized_amplitude_levels = np.linspace(0, 1, target_number_of_amplitude_levels) # Creates the target number of amplitude levels equally spaced between 0 and 1

    quantized_magnitude = np.digitize(normalized_magnitude, quantized_amplitude_levels) - 1 # Maps each continuous normalized magnitude value to a discrete index 0

    quantized_magnitude = quantized_amplitude_levels[quantized_magnitude] # Replaces each index with the actual quantized level

    quantized_magnitude *= reduced_magnitude.max() # Restores the original scale by multiplying the quantized normalized values back by the previous maximum magnitude