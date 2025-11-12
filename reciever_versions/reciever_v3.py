# --- Imports ---
import cv2
import numpy as np
import time

# --- Configuration ---
sender_output_width = 1920
sender_output_height = 1200

reference_image_seed = 42
reference_image_duration = 2.0

fps = 30

ROWS = 8
COLUMNS = 8

bit_time = 0.5  # time per bit
sync_color = (0, 255, 0)
delimiter_color = (0, 0, 255)

# --- Camera setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

# --- Helper functions ---

def generate_reference_image(seed=reference_image_seed, width=sender_output_width, height=sender_output_height):
    rng = np.random.RandomState(seed)
    reference_image = rng.randint(0, 256, (height, width), dtype=np.uint8)
    return reference_image

def read_cell_color(cell):
    """Determine dominant color in a cell"""
    hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
    # color thresholds
    red_mask = cv2.inRange(hsv, (0,100,100), (10,255,255)) | cv2.inRange(hsv, (160,100,100), (179,255,255))
    white_mask = cv2.inRange(hsv, (0,0,200), (180,30,255))
    black_mask = cv2.inRange(hsv, (0,0,0), (180,255,50))

    red_count = cv2.countNonZero(red_mask)
    white_count = cv2.countNonZero(white_mask)
    black_count = cv2.countNonZero(black_mask)

    if red_count > white_count and red_count > black_count:
        return "red"
    elif white_count > black_count:
        return "white"
    else:
        return "black"

def find_screen(frame, reference):
    """Locate the sender screen in the camera frame using template matching"""
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(frame_gray, reference, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    x, y = max_loc
    h, w = reference.shape
    return x, y, w, h

def deskew_screen(screen):
    """Estimate rotation and correct it using contours"""
    gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    coords = np.column_stack(np.where(thresh > 0))
    angle = 0
    if len(coords) > 0:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

    (h, w) = screen.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(screen, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def extract_bits(screen_frame):
    """Split the screen into ROWS x COLUMNS and read bits"""
    bits = ""
    h, w, _ = screen_frame.shape
    cell_h = h // ROWS
    cell_w = w // COLUMNS

    for r in range(ROWS):
        for c in range(COLUMNS):
            cell = screen_frame[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            color = read_cell_color(cell)
            bit = "1" if color == "white" else "0"
            bits += bit
    return bits

# --- Main loop ---
def receive_message():
    reference_image = generate_reference_image()
    reference_bgr = cv2.cvtColor(reference_image, cv2.COLOR_GRAY2BGR)

    decoding = False
    bits_accum = ""
    message = ""

    print("Press 's' to start decoding or 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret or screen_frame is None: 
            print("Warning: failed to capture frame")
            continue
           

        frame = cv2.flip(frame, 1)
        frame_vis = frame.copy()

        # detect reference image region
        try:
            x, y, w, h = find_screen(frame, reference_image)
            screen_frame = frame[y:y+h, x:x+w]
            screen_frame = deskew_screen(screen_frame)
        except:
            screen_frame = frame

        # detect sync/delimiter
        center_color = read_cell_color(screen_frame[h//2-50:h//2+50, w//2-50:w//2+50])
        cv2.putText(frame_vis, f"Center color: {center_color}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        cv2.rectangle(frame_vis, (x,y), (x+w, y+h), (0,255,0), 2)

        cv2.imshow("Receiver", frame_vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            decoding = True
            print("Started decoding...")

        elif key == ord('q'):
            break

        if decoding:
            if center_color == "green":  # sync
                bits_accum = ""
            elif center_color == "red":  # delimiter
                if len(bits_accum) >= 8:
                    for i in range(0, len(bits_accum), 8):
                        byte = bits_accum[i:i+8]
                        char = chr(int(byte, 2))
                        message += char
                        print(f"Received char: {char}")
                bits_accum = ""
            else:  # black/white bits
                frame_bits = extract_bits(screen_frame)
                bits_accum += frame_bits
                print(f"Bits: {frame_bits}")

    print("Final message:", message)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    receive_message()
