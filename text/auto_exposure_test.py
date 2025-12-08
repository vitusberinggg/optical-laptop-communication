import cv2

# Open the default camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW works well on Windows

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Attempt to set exposure to 0 (may be ignored due to AE)
cap.set(cv2.CAP_PROP_EXPOSURE, 0)
print(f"CAP_PROP_EXPOSURE set to 0, driver reports: {cap.get(cv2.CAP_PROP_EXPOSURE)}")

print("Press 'q' to quit the camera feed.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display the frame
    cv2.imshow("Camera Feed", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
