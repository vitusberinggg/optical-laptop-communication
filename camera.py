# camera_viewer.py
import cv2
import time
import os

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

    # Prepare video writer
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    # Grab first frame for size
    ret, sample_frame = cap.read()
    if not ret:
        print("Failed to grab initial frame.")
        cap.release()
        return

    height, width = sample_frame.shape[:2]

    # ---- SAVE LOCATION YOU REQUESTED ----
    output_path = r"C:\Users\eanpaln\Videos\Screen Recordings\rec4.mp4"
    # -------------------------------------

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print("WARNING: VideoWriter could not be opened. Recording disabled.")
        writer = None
    else:
        print(f"Recording to {output_path} at {fps:.2f} FPS, resolution {width}x{height}")

    print("Press 'q' in the video window to quit.")
    prev = time.time()
    frames = 0

    # Show/write first frame
    frames += 1
    if writer is not None:
        writer.write(sample_frame)
    cv2.imshow("Camera View", sample_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frames += 1
        now = time.time()
        if now - prev >= 1.0:
            print(f"FPS: {frames}")
            frames = 0
            prev = now

        if writer is not None:
            writer.write(frame)

        cv2.imshow("Camera View", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved recording to: {output_path}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
