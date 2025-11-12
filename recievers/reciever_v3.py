import cv2
import numpy as np
import time

# ----- Config -----
sender_output_width = 1920
sender_output_height = 1200
reference_image_seed = 42
DETECT_THRESHOLD = 0.55   # template match threshold (tune: 0.5-0.8)
ROWS = 8
COLUMNS = 8
BIT_TIME = 0.5

# ----- Camera init -----
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Error: Cannot open camera")

# Optional: try to set a larger resolution (may not be supported)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ----- Helpers -----
def generate_reference_image(seed=reference_image_seed, w=sender_output_width, h=sender_output_height):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w), dtype=np.uint8)

def safe_match_template(frame_gray, ref):
    # ensure reference fits in the frame; if not, scale reference down preserving aspect
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
    return max_val, max_loc, ref_small.shape  # (score, (x,y), (h,w))

def read_cell_color(cell):
    # simple dominant-color detection for a region; returns 'red','white','black','green' if matches approx
    hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
    # red
    rmask = cv2.inRange(hsv, (0,80,50), (10,255,255)) | cv2.inRange(hsv, (160,80,50), (179,255,255))
    # green
    gmask = cv2.inRange(hsv, (40,80,50), (85,255,255))
    # white / black
    wmask = cv2.inRange(hsv, (0,0,200), (180,40,255))
    bmask = cv2.inRange(hsv, (0,0,0), (180,255,60))
    counts = {
        "red": int(cv2.countNonZero(rmask)),
        "green": int(cv2.countNonZero(gmask)),
        "white": int(cv2.countNonZero(wmask)),
        "black": int(cv2.countNonZero(bmask))
    }
    # return the color with highest count
    color = max(counts, key=counts.get)
    return color

def extract_bits_from_screen(screen_frame, rows=ROWS, cols=COLUMNS, threshold=127):
    gray = cv2.cvtColor(screen_frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    ch = max(1, h // rows)
    cw = max(1, w // cols)
    bits = ""
    for r in range(rows):
        for c in range(cols):
            y1 = r*ch; y2 = min(h, (r+1)*ch)
            x1 = c*cw; x2 = min(w, (c+1)*cw)
            cell = gray[y1:y2, x1:x2]
            if cell.size == 0:
                val = 0
            else:
                val = 1 if cell.mean() > threshold else 0
            bits += "1" if val else "0"
    return bits

# ----- Main receive loop -----
def receive_message():
    reference = generate_reference_image()
    reference_gray = reference  # already grayscale
    window_name = "Receiver"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    found = False
    ref_bbox = None  # (x,y,w,h)
    accumulated_bits = ""
    message = ""

    print("Starting receiver. Press 'q' to quit, 's' to start decoding manually.")
    last_sample_time = None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            # show a blank if camera fails but keep loop alive
            blank = np.zeros((480,640,3), dtype=np.uint8)
            cv2.putText(blank, "Camera read failed", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.imshow(window_name, blank)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        vis = frame.copy()

        # attempt to detect reference image on this frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score, loc, ref_shape = safe_match_template(frame_gray, reference_gray)

        if score >= DETECT_THRESHOLD:
            rx, ry = loc
            rh, rw = ref_shape
            # check bounds
            fh, fw = frame.shape[:2]
            if ry + rh <= fh and rx + rw <= fw and rh > 0 and rw > 0:
                found = True
                ref_bbox = (rx, ry, rw, rh)
                # draw rectangle where reference matched
                cv2.rectangle(vis, (rx, ry), (rx+rw, ry+rh), (0,255,0), 2)
                cv2.putText(vis, f"Ref found: {score:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            else:
                found = False
                ref_bbox = None
        else:
            # not found - show score for debugging
            cv2.putText(vis, f"Ref score: {score:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,200), 2)
            found = False
            ref_bbox = None

        # Always show camera view
        cv2.imshow(window_name, vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # if we have a valid bbox, crop and process
        if ref_bbox is None:
            continue

        x, y, w, h = ref_bbox
        screen = frame[y:y+h, x:x+w]
        if screen is None or screen.size == 0:
            # skip if empty crop
            continue

        # Optional: deskew/perspective correction can be added here if needed.
        # For safety, take a small center region and compute its dominant color
        ch = max(1, h // 8)
        cw = max(1, w // 8)
        cy1 = max(0, h//2 - ch//2); cy2 = min(h, h//2 + ch//2)
        cx1 = max(0, w//2 - cw//2); cx2 = min(w, w//2 + cw//2)
        center_cell = screen[cy1:cy2, cx1:cx2]
        if center_cell.size == 0:
            continue
        center_color = read_cell_color(center_cell)

        # Handle sync/delimiter/data
        if center_color == "green":
            # sync: reset accumulator
            accumulated_bits = ""
            print("[sync] detected, accumulator cleared")
            # wait a short time so we don't repeatedly reset
            time.sleep(0.05)
            continue
        elif center_color == "red":
            # delimiter: convert accumulated bits into bytes
            if len(accumulated_bits) > 0:
                for i in range(0, len(accumulated_bits), 8):
                    byte = accumulated_bits[i:i+8]
                    if len(byte) < 8:
                        byte = byte.ljust(8, '0')
                    try:
                        ch_val = chr(int(byte, 2))
                    except Exception:
                        ch_val = '?'
                    message += ch_val
                    print("Received char:", ch_val)
            accumulated_bits = ""
            time.sleep(0.05)
            continue
        else:
            # data frame: sample a window of time equal to BIT_TIME and aggregate frames
            now = time.time()
            if last_sample_time is None:
                last_sample_time = now

            # for a simple first pass: wait BIT_TIME then sample the current screen only
            if now - last_sample_time < BIT_TIME:
                # you can do aggregation (collect multiple frames) here for robustness
                continue
            last_sample_time = now

            # read bits from the screen crop
            bits = extract_bits_from_screen(screen, ROWS, COLUMNS)
            accumulated_bits += bits
            print("Read bits:", bits)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    receive_message()
