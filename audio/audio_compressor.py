
# --- Imports ---

import numpy as np # Library for handling numerical arrays, needed to manipulate matrices
import librosa # High-level audio processing library to load audio files, compute frequency spectrograms, resample etc.
import soundfile

# --- Definitions ---

target_number_of_frequencies = 8 # The number of frequencies we reduce the spectrograms to (lower amount --> fewer frequency details --> less data)
target_number_of_amplitude_levels = 12 # The number of amplitude levels we keep per frequency bin (fewer levels --> coarser dynamics)

hop_length = 512 # The amount of samples between each spectrogram frame (shorter hop length --> more overlap between windows --> smoother time reconstruction but more computation)

target_sample_rate = 8000 # The audio signal's average number of samples (values) per second

seconds_per_time_frame = hop_length / target_sample_rate # Duration of one STFT hop

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

    print(f"\n[INFO] Reducing the amount of frequency bins...")

    reduced_magnitude = np.zeros_like(spectrogram_magnitude)

    number_of_time_frames = spectrogram_magnitude.shape[1]

    for time_frame in range(number_of_time_frames):

        column = spectrogram_magnitude[:, time_frame]

        loudest_frequency_indices = np.argsort(column)[-target_number_of_frequencies:]

        reduced_magnitude[loudest_frequency_indices, time_frame] = column[loudest_frequency_indices]

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

        expanded_magnitude = quantized_magnitude
        
        reconstructed_spectrogram = expanded_magnitude * np.exp(1j * spectrogram_phase) # Reconstructs the spectrogram based on the magnitude matrix

        reconstructed_audio_signal = librosa.istft(reconstructed_spectrogram, hop_length = hop_length) # Inverse transformation based on spectrogram

        output_file = "compressed_audio.wav"

        soundfile.write(output_file, reconstructed_audio_signal, target_sample_rate)

        print(f"\n[INFO] Compression done. Compressed audio file path: {output_file}")
    
    else:
        print("\n[INFO] Compression done.")

    # Bitrate calculation

    print("\n[INFO] Calculating bitrate...")

    number_of_frequency_bins = spectrogram_magnitude.shape[0]

    total_duration = number_of_time_frames * seconds_per_time_frame # Total reconstructed audio duration

    print(f"\n[INFO] Audio duration: {total_duration:.2f} s")

    bits_per_frequency_index = int(np.ceil(np.log2(number_of_frequency_bins))) # Bits required to store frequency index

    print(f"\n[INFO] Bits per frequency index: {bits_per_frequency_index} bits")

    bits_per_amplitude_level = int(np.ceil(np.log2(target_number_of_amplitude_levels))) # Bits required to store quantized amplitude level

    print(f"\n[INFO] Bits per amplitude level: {bits_per_amplitude_level} bits")

    bits_per_frame = target_number_of_frequencies * (bits_per_frequency_index + bits_per_amplitude_level)

    total_amount_of_bits = bits_per_frame * number_of_time_frames

    print(f"\n[INFO] Total amount of bits: {total_amount_of_bits}")

    bitrate = total_amount_of_bits / total_duration

    print(f"\n[INFO] Bitrate: {round(bitrate)} bits/s")

# --- Execution ---

if __name__ == "__main__":
    audio_compressor(input_file)