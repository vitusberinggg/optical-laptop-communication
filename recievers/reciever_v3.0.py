import cv2
import numpy as np
import time

# ----- Config -----
SENDER_WIDTH = 1920
SENDER_HEIGHT = 1200
REFERENCE_SEED = 42
DETECT_THRESHOLD = 0.55   # template match threshold
ROWS = 8                   # number of rows in the bit grid
COLS = 8                   # number of columns in the bit grid
BIT_TIME = 0.5             # seconds per frame
CENTER_CHECK_SIZE = 0.08   # fraction of screen for center check (green/red)
FRAME_AVG_COUNT = 3        # number of frames to average per data frame

# ----- Camera init -----
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Error: Cannot open camera")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ----- Helpers -----
def generate_reference_image(seed=REFERENCE_SEED, w=SENDER_WIDTH, h=SENDER_HEIGHT):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w), dtype=np.uint8)

def safe_match_template(frame_gray, ref):
    fh, fw = frame_gray.shape
    rh, rw = ref.shape
    if rh > fh or rw > fw:
        scale = min(fw / rw, fh / rh, 1.0)
        ref = cv2.resize(ref, (int(rw*scale), int(rh*scale)), interpolation=cv2.INTER_AREA)
    res = cv2.matchTemplate(frame_gray, ref, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return max_val, max_loc, ref.shape

def extract_bits_from_frame(frame, rows=ROWS, cols=COLS, threshold=127):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    bits = ""
    for r in range(rows):
        for c in range(cols):
            cell = gray[r*h//rows:(r+1)*h//rows, c*w//cols:(c+1)*w//cols]
            val = 1 if cell.mean() > threshold else 0
            bits += "1" if val else "0"
    return bits

def is_end_frame(frame):
    avg_color = frame.mean(axis=(0,1))
    return avg_color[2] > 200 and avg_color[0] < 50 and avg_color[1] < 50

def read_center_color(frame, center_fraction=CENTER_CHECK_SIZE):
    h, w = frame.shape[:2]
    cy1 = int(h*(0.5 - center_fraction/2))
    cy2 = int(h*(0.5 + center_fraction/2))
    cx1 = int(w*(0.5 - center_fraction/2))
    cx2 = int(w*(0.5 + center_fraction/2))
    cell = frame[cy1:cy2, cx1:cx2]
    if cell.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
    rmask = cv2.inRange(hsv, (0,80,50), (10,255,255)) | cv2.inRange(hsv, (160,80,50), (179,255,255))
    gmask = cv2.inRange(hsv, (40,80,50), (85,255,255))
    wmask = cv2.inRange(hsv, (0,0,200), (180,40,255))
    bmask = cv2.inRange(hsv, (0,0,0), (180,255,60))
    counts = {"red": int(cv2.countNonZero(rmask)),
              "green": int(cv2.countNonZero(gmask)),
              "white": int(cv2.countNonZero(wmask)),
              "black": int(cv2.countNonZero(bmask))}
    return max(counts, key=counts.get)

def straighten_screen(frame, pts_src, width=SENDER_WIDTH, height=SENDER_HEIGHT):
    pts_dst = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts_src.astype(np.float32), pts_dst)
    return cv2.warpPerspective(frame, M, (width, height))

# ----- Receiver -----
def receive_message():
    reference = generate_reference_image()
    reference_gray = reference
    window_name = "Receiver"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    accumulated_bits = ""
    message = ""
    ref_bbox = None
    screen_pts = None

    print("Waiting for sender reference image...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        vis = frame.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Reference detection ---
        if screen_pts is None:
            score, loc, ref_shape = safe_match_template(frame_gray, reference_gray)
            if score >= DETECT_THRESHOLD:
                x, y = loc
                rh, rw = ref_shape
                ref_bbox = (x, y, rw, rh)
                cv2.rectangle(vis, (x, y), (x+rw, y+rh), (0,255,0), 2)
                cv2.putText(vis, f"Ref found: {score:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                # Define the four corners for perspective correction
                screen_pts = np.array([[x,y],[x+rw,y],[x+rw,y+rh],[x,y+rh]], dtype=np.float32)
                print("[SYNC] Reference detected, straightening screen...")

        # --- Once reference detected ---
        if screen_pts is not None:
            straight = straighten_screen(frame, screen_pts)

            # --- Check center color ---
            center_color = read_center_color(straight)
            if center_color == "red":
                # finalize message
                frame_bits = extract_bits_from_frame(straight)
                accumulated_bits += frame_bits

                decoded_chars = []
                for i in range(0, len(accumulated_bits), 8):
                    byte = accumulated_bits[i:i+8]
                    if len(byte)<8:
                        byte = byte.ljust(8,'0')
                    try:
                        decoded_chars.append(chr(int(byte,2)))
                    except:
                        decoded_chars.append('?')
                message = "".join(decoded_chars)
                print("=== Final Message ===")
                print(message)
                break

            # --- Capture multiple frames for averaging ---
            frames = [straight]
            t0 = time.time()
            while len(frames) < FRAME_AVG_COUNT and (time.time()-t0)<BIT_TIME:
                ret2, f2 = cap.read()
                if not ret2:
                    continue
                straight2 = straighten_screen(f2, screen_pts)
                frames.append(straight2)

            avg_frame = np.mean(np.stack(frames, axis=0).astype(np.float32), axis=0).astype(np.uint8)
            frame_bits = extract_bits_from_frame(avg_frame)
            accumulated_bits += frame_bits
            print("Read bits:", frame_bits)

        cv2.imshow(window_name, vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Receiver finished.")

if __name__ == "__main__":
    receive_message()
