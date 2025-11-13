
# --- Imports ---

import cv2
import numpy as np
import time

from utilities.generate_reference_image import generate_reference_image
from utilities.global_definitions import (
    sender_output_height,
    sender_output_width,
    reference_image_duration,
    frame_duration,
    number_of_columns,
    number_of_rows,
    end_frame_color
)

# --- Definitions ---

brightness_threshold = 100 # Brightness threshold for determining bit values

camera_index = 0 # Index of the camera to be used

end_frame_detection_tolerance = 40 # Tolerance for end frame color detection