# camera_viewer.py
import cv2
import time

def main():
    # Try DirectShow on Windows (works well); remove second arg if you're on Linux/macOS
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    except Exception:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    # Optional: set a desired resolution (comment out if undesired)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)

    # ---- FULLSCREEN WINDOW ----
    cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Camera View", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # ---------------------------

    print("Press 'q' in the video window to quit.")
    prev = time.time()
    frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frames += 1
        now = time.time()
        if now - prev >= 1.0:
            # print FPS once per second
            print(f"FPS: {frames}")
            frames = 0
            prev = now

        cv2.imshow("Camera View", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
