
# --- Imports ---

# Library imports

import numpy as np # Library for handling numerical arrays, needed to manipulate matrices
import librosa # High-level audio processing library to load audio files, compute frequency spectrograms, resample etc.
import soundfile

# --- Definitions ---

"""

"frequency_spectrogram_frame_size" and "hop_length" trade time vs frequency resolution.
For short, percussive sounds, smaller frames and smaller hops preserve attacks better.
For tonal music, larger frames give more stable pitch bins.

"""

frequency_bin_reduction_method = "spectral-sparsification"

target_number_of_frequencies = 32 # The number of frequencies we reduce the spectrograms to (lower amount --> fewer frequency details --> less data)
target_number_of_amplitude_levels = 3 # The number of amplitude levels we keep per frequency bin (fewer levels --> coarser dynamics)

hop_length = 256 # The amount of samples between each spectrogram frame (shorter hop length --> more overlap between windows --> smoother time reconstruction but more computation)

target_sample_rate = 8000 # The audio signal's average number of samples (values) per second

frequency_spectrogram_frame_size = 1024 # The amount of samples each spectogram contains (larger windows give higher frequency resolution but lower time resolution)

input_file = "The Chords - Sh-Boom.mp3"

save_compressed_file = True

# --- Main function ---

def audio_compressor(input_file):

    """
    Heavily compresses an audio file by removing frequencies and amplitudes, only keeping the bare minimum for the audio to still be recognizable.

    Arguments:
        "input_file" (str): The audio file to compress.

    Returns:
        None
    
    """

    # Spectrogram creation

    print("\n[INFO] Creating a spectrogram...")
    
    audio_signal, _ = librosa.load(input_file, sr = target_sample_rate, mono = True) # Reads the audio file

    spectrogram = librosa.stft(audio_signal, n_fft = frequency_spectrogram_frame_size, hop_length = hop_length) # Computes the STFT, which converts the time-domain signal into a complex matrix where each row corresponds to a frequency bin, and each column corresponds to a time frame

    spectrogram_magnitude = np.abs(spectrogram) # The absolute value (amplitude) of each frequency bin at each time frame
    spectrogram_phase = np.angle(spectrogram) # The phase of each frequency bin (needed for exact waveform reconstruction, as quantizing causes artifacts)

    print("\n[INFO] Spectrogram created.")

    # Frequency bin reduction

    print("\n[INFO] Reducing the amount of frequency bins...")

    number_of_frequency_bins = spectrogram_magnitude.shape[0] # Gets the original number of frequency bins

    frequency_bin_factor = number_of_frequency_bins // target_number_of_frequencies

    if frequency_bin_reduction_method == "spectral-sparsification":

        reduced_magnitude = np.zeros_like(spectrogram_magnitude)

        number_of_time_frames = spectrogram_magnitude.shape[1]

        for time_frame in range(number_of_time_frames):

            column = spectrogram_magnitude[:, time_frame]

            loudest_frequency_indices = np.argsort(column)[-target_number_of_frequencies:]

            reduced_magnitude[loudest_frequency_indices, time_frame] = column[loudest_frequency_indices]
    
    else:
        reduced_magnitude = spectrogram_magnitude[:frequency_bin_factor * target_number_of_frequencies].reshape(target_number_of_frequencies, frequency_bin_factor, -1).mean(axis = 1)

    if reduced_magnitude.max() != 0: # If the input isn't silent:
        normalized_magnitude = reduced_magnitude / reduced_magnitude.max() # Normalizes the magnitude by dividing it by its maximum value to ensure that quantization levels are interpreted relative to the maximum energy present

    else: # Else (if it's completely silent):
        normalized_magnitude = reduced_magnitude # (To avoid division by zero)

    print(f"\n[INFO] Reduced the amount of frequency bins to {target_number_of_frequencies}")

    # Amplitude quantization

    print("\n[INFO] Quantizing the amplitude...")

    quantized_amplitude_levels = np.linspace(0, 1, target_number_of_amplitude_levels) # Creates the target number of amplitude levels equally spaced between 0 and 1

    quantized_magnitude = np.digitize(normalized_magnitude, quantized_amplitude_levels) - 1 # Maps each continuous normalized magnitude value to a discrete index 0

    quantized_magnitude = quantized_amplitude_levels[quantized_magnitude] # Replaces each index with the actual quantized level

    quantized_magnitude *= reduced_magnitude.max() # Restores the original scale by multiplying the quantized normalized values back by the previous maximum magnitude

    print(f"\n[INFO] Reduced the amount of amplitude levels to {target_number_of_amplitude_levels}")

    # Inverse transform

    if save_compressed_file:

        if frequency_bin_reduction_method == "spectral-sparsification":
            expanded_magnitude = quantized_magnitude
        
        else:

            expanded_magnitude = np.repeat(quantized_magnitude, frequency_bin_factor, axis = 0) # Repeats every coarse frequency row "frequency_bin_factor" times to make the reduced magnitude matrix return to the original number of frequency rows expected by the ISTFT

            if expanded_magnitude.shape[0] < number_of_frequency_bins: # If the expanded magnitude matrix still has fewer rows than expected:

                number_of_padding_rows = number_of_frequency_bins - expanded_magnitude.shape[0]

                expanded_magnitude = np.vstack([expanded_magnitude, np.zeros((number_of_padding_rows, expanded_magnitude.shape[1]))]) # Pad the top with zeros to make shapes match (removes high frequency content that didn't fit into complete groups)
        
        reconstructed_spectrogram = expanded_magnitude * np.exp(1j * spectrogram_phase) # Reconstructs the spectrogram based on the magnitude matrix

        reconstructed_audio_signal = librosa.istft(reconstructed_spectrogram, hop_length = hop_length) # Inverse transformation based on spectrogram

        output_file = "compressed_audio.wav"

        soundfile.write(output_file, reconstructed_audio_signal, target_sample_rate)

        print(f"\n[INFO] Compression done. Compressed audio file path: {output_file}")
    
    else:
        print("\n[INFO] Compression done.")

# --- Execution ---

if __name__ == "__main__":
    audio_compressor(input_file)