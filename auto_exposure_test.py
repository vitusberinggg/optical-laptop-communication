import cv2

# Open the camera (0 = default webcam)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW often works better on Windows

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Try to disable auto-exposure
# Note: driver behavior varies; some expect 0,1, 0.25, 0.75, or 3 for auto-exposure
auto_exp_values = [1, 0, 0.25, 0.75, 3]
for val in auto_exp_values:
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, val)
    print(f"Set CAP_PROP_AUTO_EXPOSURE = {val}, driver reports: {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")

# Try to set manual exposure value (if AE can be disabled)
# Units vary by driver; negative log2 exposure values are common
manual_exposure_values = [-8, -6, -4, -2, 0]
for exp in manual_exposure_values:
    cap.set(cv2.CAP_PROP_EXPOSURE, exp)
    print(f"Set CAP_PROP_EXPOSURE = {exp}, driver reports: {cap.get(cv2.CAP_PROP_EXPOSURE)}")

# Optionally, try disabling auto-white-balance
cap.set(cv2.CAP_PROP_AUTO_WB, 0)
print(f"CAP_PROP_AUTO_WB = {cap.get(cv2.CAP_PROP_AUTO_WB)}")

# Optional: disable autofocus
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
print(f"CAP_PROP_AUTOFOCUS = {cap.get(cv2.CAP_PROP_AUTOFOCUS)}")

print("Press 'q' to exit the video window")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show the camera feed
    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
