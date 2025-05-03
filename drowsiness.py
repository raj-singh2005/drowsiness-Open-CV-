
import cv2

# Load Haar cascades for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Start webcam
cap = cv2.VideoCapture(0)

# Frame thresholds
drowsy_frames_threshold = 20
drowsy_counter = 0
eyes_detected_consecutively = 0

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    success, img = cap.read()
    if not success:
        break

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(imgGray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Detect eyes only in the top half of the face
        roi_gray = imgGray[y:y + h // 2, x:x + w]
        roi_color = img[y:y + h // 2, x:x + w]

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(30, 30)
        )

        print(f"Eyes detected: {len(eyes)}")  # Debug info

        if len(eyes) < 2:
            drowsy_counter += 1
            eyes_detected_consecutively = 0
            if drowsy_counter >= drowsy_frames_threshold:
                cv2.putText(img, "DROWSY", (50, 50), font, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            eyes_detected_consecutively += 1
            if eyes_detected_consecutively >= 3:
                drowsy_counter = 0  # Only reset after a few good frames

            # Draw eye rectangles
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Show result
    cv2.imshow('Drowsiness Detection', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()