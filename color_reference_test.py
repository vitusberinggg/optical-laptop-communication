import numpy as np
import cv2

from utilities.global_definitions import (
    red_bgr, green_bgr, blue_bgr
)
from utilities.color_functions import color_offset_calculation
from utilities.image_generation_functions import create_color_reference_frame

frame = create_color_reference_frame()

corrected_ranges = color_offset_calculation(frame)

print("\nCorrected HSV Ranges:")
for color, (lower, upper) in corrected_ranges.items():
    print(f"{color}: lower={lower}, upper={upper}")
