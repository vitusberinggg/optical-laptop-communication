import cv2
import numpy as np
import time

# ----- Config -----
sender_output_width = 1920
sender_output_height = 1200
reference_image_seed = 42
DETECT_THRESHOLD = 0.55   # template match threshold
ROWS = 8                   # number of rows in the data grid
COLUMNS = 8                # number of columns in the data grid
BIT_TIME = 0.5             # seconds to capture multiple frames per data bit
CENTER_CHECK_SIZE = 0.08   # proportion of screen used for center color checks

# ----- Camera init -----
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Error: Cannot open camera")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ----- Helper Functions -----
def generate_reference_image(seed=reference_image_seed, w=sender_output_width, h=sender_output_height):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w), dtype=np.uint8)

def safe_match_template(frame_gray, ref):
    fh, fw = frame_gray.shape
    rh, rw = ref.shape
    if rh > fh or rw > fw:
        scale = min(fw / rw, fh / rh, 1.0)
        new_w = max(1, int(rw * scale))
        new_h = max(1, int(rh * scale))
        ref_small = cv2.resize(ref, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        ref_small = ref
    res = cv2.matchTemplate(frame_gray, ref_small, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return max_val, max_loc, ref_small.shape

def read_cell_color(cell):
    if cell is None or cell.size == 0:
        return "black"
    hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
    rmask = cv2.inRange(hsv, (0,80,50), (10,255,255)) | cv2.inRange(hsv, (160,80,50), (179,255,255))
    gmask = cv2.inRange(hsv, (40,80,50), (85,255,255))
    wmask = cv2.inRange(hsv, (0,0,200), (180,40,255))
    bmask = cv2.inRange(hsv, (0,0,0), (180,255,60))
    counts = {
        "red": int(cv2.countNonZero(rmask)),
        "green": int(cv2.countNonZero(gmask)),
        "white": int(cv2.countNonZero(wmask)),
        "black": int(cv2.countNonZero(bmask))
    }
    return max(counts, key=counts.get)

def extract_bits_from_avg_frame(avg_frame, rows=ROWS, cols=COLUMNS, threshold=127):
    gray = cv2.cvtColor(avg_frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    ch = max(1, h // rows)
    cw = max(1, w // cols)
    bits = ""
    for r in range(rows):
        for c in range(cols):
            y1, y2 = r*ch, min(h, (r+1)*ch)
            x1, x2 = c*cw, min(w, (c+1)*cw)
            cell = gray[y1:y2, x1:x2]
            val = 1 if cell.mean() > threshold else 0 if cell.size else 0
            bits += "1" if val else "0"
    return bits

def majority_bits_from_frames(frames, rows=ROWS, cols=COLUMNS, threshold=127):
    if not frames:
        return "0"*(rows*cols)
    avg = np.mean(np.stack(frames, axis=0).astype(np.float32), axis=0).astype(np.uint8)
    return extract_bits_from_avg_frame(avg, rows, cols, threshold)

# ----- Main Receiver -----
def receive_message():
    reference = generate_reference_image()
    reference_gray = reference
    window_name = "Receiver"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    ref_bbox = None
    accumulated_bits = ""
    message = ""
    stop_flag = False

    print("Receiver started. Show the sender reference image to the camera.")
    print("Press 'Q' at any time to quit.")

    while not stop_flag:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        vis = frame.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score, loc, ref_shape = safe_match_template(frame_gray, reference_gray)

        if score >= DETECT_THRESHOLD:
            rx, ry = loc
            rh, rw = ref_shape
            fh, fw = frame.shape[:2]
            if ry + rh <= fh and rx + rw <= fw:
                ref_bbox = (rx, ry, rw, rh)
                cv2.rectangle(vis, (rx, ry), (rx+rw, ry+rh), (0,255,0), 2)
                cv2.putText(vis, f"Ref found: {score:.2f}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            else:
                ref_bbox = None
        else:
            ref_bbox = None

        cv2.imshow(window_name, vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_flag = True
            break

        if ref_bbox is None:
            continue

        x, y, w, h = ref_bbox
        screen = frame[y:y+h, x:x+w]

        cy1 = int(max(0, h*(0.5-CENTER_CHECK_SIZE/2)))
        cy2 = int(min(h, h*(0.5+CENTER_CHECK_SIZE/2)))
        cx1 = int(max(0, w*(0.5-CENTER_CHECK_SIZE/2)))
        cx2 = int(min(w, w*(0.5+CENTER_CHECK_SIZE/2)))

        # --- Capture frames for BIT_TIME window ---
        window_frames = []
        t_start = time.time()
        while time.time() - t_start < BIT_TIME:
            ret2, f2 = cap.read()
            if not ret2:
                continue
            screen2 = f2[y:y+h, x:x+w]
            window_frames.append(screen2.copy())

            k2 = cv2.waitKey(1) & 0xFF
            if k2 == ord('q'):
                stop_flag = True
                break
        if stop_flag:
            break

        avg_frame = np.mean(np.stack(window_frames, axis=0).astype(np.float32), axis=0).astype(np.uint8)
        center_cell = avg_frame[cy1:cy2, cx1:cx2]
        center_color = read_cell_color(center_cell)

        if center_color == "green":
            accumulated_bits = ""
            print("[sync] cleared")
            time.sleep(0.05)
            continue

        if center_color == "red":
            frame_bits = majority_bits_from_frames(window_frames, ROWS, COLUMNS)
            accumulated_bits += frame_bits

            # decode accumulated bits to chars
            decoded_chars = []
            for i in range(0, len(accumulated_bits), 8):
                byte = accumulated_bits[i:i+8]
                if len(byte) < 8:
                    byte = byte.ljust(8,'0')
                try:
                    decoded_chars.append(chr(int(byte,2)))
                except:
                    decoded_chars.append('?')
            decoded_message = "".join(decoded_chars)
            print("=== Decoded message ===")
            print(decoded_message)
            print("=======================")
            accumulated_bits = ""
