import cv2

cap = cv2.VideoCapture("C:\Users\eanpaln\Videos\Screen Recordings\Recordinggg.mp4")
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Optional: resize
    frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))

    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    cv2.imshow("Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
