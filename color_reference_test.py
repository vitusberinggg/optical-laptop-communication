import numpy as np
import cv2

from utilities.global_definitions import (
    red_bgr, green_bgr, blue_bgr
)
from utilities.color_functions_v3 import color_offset_calculation
from utilities.image_generation_functions import create_color_reference_frame

frame = create_color_reference_frame()

corrected_ranges = color_offset_calculation(frame)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
thickness = 1

stripe_width = frame.shape[1] // 3
colors = ["red1","red2", "green", "blue"]

for i, color in enumerate(colors):
    x_start = i * stripe_width + 5  
    y_start = 30  
    lower, upper = corrected_ranges[color]
    text = f"{color}: lower={lower}, upper={upper}"
    cv2.putText(frame, text, (x_start, y_start), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

cv2.imshow("Color Reference Frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nCorrected HSV Ranges:")
for color, (lower, upper) in corrected_ranges.items():
    print(f"{color}: lower={lower}, upper={upper}")
