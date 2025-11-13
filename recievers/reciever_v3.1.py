
# --- Imports ---

import cv2
import numpy as np
import time

from utilities.generate_reference_image import generate_reference_image
from utilities.detect_end_frame import detect_end_frame
from utilities.detect_reference_image import detect_reference_image
from utilities.decode_bitgrid import decode_bitgrid
from utilities.bits_to_message import bits_to_message
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

camera_index = 0 # Index of the camera to be used

# --- Main function ---

def receive_message():

    """
    Receives a message from the sender.
    
    Arguments:
        None
        
    Returns:
        None
    
    """

    videoCapture = cv2.VideoCapture(camera_index)
    videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, sender_output_width)
    videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, sender_output_height)

    reference_image = generate_reference_image()

    print("[INFO] Waiting for the reference frame...")

    reference_image_detected = False
    reference_start_time = None

    bitstring = ""
    frame_counter = 0
    end_frame_detected = False

    while True:

        ret, frame = videoCapture.read()

        if not ret:
            print("[WARN] Failed to grab frame.")
            continue

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_display = frame.copy()

        if not reference_image_detected:

            detected, correlation = detect_reference_image(frame_gray, reference_image)

            cv2.putText(frame_display, f"Ref Corr: {correlation:.3f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if detected:
                reference_image_detected = True
                reference_start_time = time.time()
                print("[INFO] Reference image detected — syncing started.")

        else:

            if time.time() - reference_start_time > reference_image_duration:
                
                if detect_end_frame(frame):
                    print("[INFO] End frame detected. Stopping capture.")
                    end_detected = True
                    break

                bits = decode_bitgrid(frame_gray)
                bitstring += bits
                frame_counter += 1
                print(f"[FRAME {frame_counter}] Bits: {bits}")

        cv2.imshow("RECEIVER", frame_display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    videoCapture.release()
    cv2.destroyAllWindows()

    if end_frame_detected: # If the end frame is detected:
        message = bits_to_message(bitstring) # Decode the bitstrig
        print("\n--- Received Message ---")
        print(message)

    else:
        print("\n[INFO] No end frame detected — transmission may be incomplete.")

# --- Execution ---

if __name__ == "__main__": # If the script is run directly:
    receive_message() # Call the main function to receive the message