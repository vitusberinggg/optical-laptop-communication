
# --- Imports ---

import cv2
import math
import numpy as np

# --- BGR definitions ---

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

delta_h = 5
delta_c = 15
delta_v = 15

# --- Color maps ---

color_map_2bit = [
    (0, 0, 0),       # 0b00 = Black
    (255, 255, 255), # 0b01 = White
    (255, 0, 0),     # 0b10 = Blue 
    (0, 255, 0),     # 0b11 = Green
]

color_map_3bit = [
    black_bgr,
    white_bgr,
    red_bgr,
    green_bgr,
    blue_bgr,
    yellow_bgr,
    cyan_bgr,
    magenta_bgr
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

# Seven color calibration
#"""
idx_to_3bit = {
    0: 0b001,  # white
    1: 0b000,  # black
    2: 0b010,  # red1
    3: 0b011,  # green
    4: 0b100,  # blue
    5: 0b101,  # yellow
    6: 0b110,  # cyan
    7: 0b111,  # magenta
}
#"""

# Three color calibration
"""
idx_to_3bit = {
    0: 0b001,  # white
    1: 0b000,  # black
    2: 0b010,  # red1
    3: 0b010,  # red2
    4: 0b011,  # green
    5: 0b100,  # blue
    6: 0b101,  # yellow
    7: 0b110,  # cyan
    8: 0b111,  # magenta
}
"""
# --- Sender output definitions ---

sender_output_width = 1920 # Width of the sender output in pixels
sender_output_height = 1200 # Height of the sender output in pixels

number_of_columns = 16 # Number of columns in the frame
number_of_rows = 16 # Number of rows in the frame

number_of_cells = number_of_columns * number_of_rows

bit_cell_width = sender_output_width // number_of_columns # Width of each bit cell in pixels
bit_cell_height = sender_output_height // number_of_rows # Height of each bit cell in pixels

bits_per_cell = 3
number_of_colors = bits_per_cell^2 

frame_duration = 0.3 # Duration for each frame in seconds

orange_time = 0.2

margin = 15

message = "Hej! Jag hoppas att du har en stund för dig själv just nu, där du kan sitta ner och andas lite. Ibland känns livet som en snabb film där scenerna bara rusar förbi, och man hinner knappt hämta andan innan nästa moment kommer. Men det är viktigt att påminna sig själv om att även i de mest hektiska stunderna finns det små ögonblick av lugn, små stunder där allt känns lite mer balanserat. Tänk dig att varje dag är som ett tomt blad. Vissa dagar fylls det med färger, skratt, kanske någon tår eller två, och det är helt okej. Livet behöver alla nyanser för att bli komplett. Det du upplever, även de små, vardagliga sakerna, är viktiga delar i din historia. De är de små penseldragen som gör bilden av dig själv unik och värdefull. Jag vill också säga, ge dig själv erkännande. Allt du har klarat av hittills, alla problem du har löst, alla gånger du har rest dig upp igen efter motgångar det är styrka. Ibland är vi våra egna hårdaste kritiker, och det är lätt att glömma hur långt vi faktiskt har kommit. Men varje steg du tar, oavsett hur litet det känns, är betydelsefullt. Och vet du vad? Det är okej att inte alltid ha svaren. Det är okej att tveka, att känna sig osäker, eller att bara vilja stanna upp en stund. Ibland är det just i pauserna, i stillheten, som de bästa idéerna och de djupaste insikterna dyker upp. Så ta den där extra koppen te, gå en liten promenad, eller bara titta ut genom fönstret och observera världen en stund. Livet är fullt av små mirakel, och även om vi ibland inte ser dem direkt, finns de där, tysta men stadiga. Kom ihåg att du inte är ensam, och även när allt känns tungt finns det alltid möjligheter att hitta ljuset igen. Varje dag är en chans att skapa något nytt, att lära dig något om dig själv, och att ta hand om dig själv. Och det är något att vara stolt över. Så, avslutningsvis: fortsätt vara nyfiken, fortsätt drömma stort, och kom ihåg att ge dig själv samma vänlighet som du så lätt ger till andra. Du är värdefull, och världen är bättre med dig i den."

# --- Reciever input definitions ---

laptop_webcam_pixel_height = 1440
laptop_webcam_pixel_width = 2560

# --- ArUco marker definitions ---

aruco_marker_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_detector_parameters = cv2.aruco.DetectorParameters()

aruco_marker_margin = 15

small_aruco_marker_side_length = sender_output_height // 2 - 50
large_aruco_marker_side_length = sender_output_height - 2 * aruco_marker_margin
large_aruco_marker_side_length_without_margin = sender_output_height
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
minimized_roi_fraction = 1/12

# --- Steps definitions ---

end_bit_steps = 2
dominant_color_steps = 4

# --- Audio definitions ---

audio_file = "audio_files/The Chords - Sh-Boom_2000Hz_8_bit-PCM_Mono.wav"
#audio_file = "audio_files/The Chords - Sh-Boom.mp3"

number_of_frequencies = 50 # The number of frequencies we reduce the spectrograms to (lower amount --> fewer frequency details --> less data)
number_of_amplitude_levels = 12 # The number of amplitude levels we keep per frequency bin (fewer levels --> coarser dynamics)

quantized_amplitude_levels = np.linspace(0.01, 1, number_of_amplitude_levels) # Creates the target number of amplitude levels equally spaced between 0 and 1

bits_per_frequency = int(math.log2(number_of_frequencies))
bits_per_amplitude_level = int(math.log2(bits_per_frequency))

bits_per_audio_time_frame = number_of_frequencies * (bits_per_frequency + bits_per_amplitude_level)

bits_per_visual_frame = number_of_cells * bits_per_cell

hop_length = 512 # The amount of samples between each spectrogram frame (shorter hop length --> more overlap between windows --> smoother time reconstruction but more computation)

sample_rate = 2000 # The audio signal's average number of samples (values) per second

seconds_per_time_frame = hop_length / sample_rate # Duration of one STFT hop

frequency_spectrogram_frame_size = 1024 # The amount of samples each spectogram contains (larger windows give higher frequency resolution but lower time resolution)