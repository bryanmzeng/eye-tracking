import cv2
from facenet_pytorch import MTCNN

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True)

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Detect faces using MTCNN
    boxes, probs = mtcnn.detect(frame)

    # Check if faces are detected
    if boxes is not None:
        for box in boxes:
            x, y, w, h = map(int, box)

            # Region of Interest (ROI) for eyes in the face
            roi_gray = frame[y:y+h, x:x+w]

            # Use Haarcascades for eye detection
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(roi_gray)

            # Sort eyes by area in descending order
            eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]

            # Iterate over the top two detected eyes
            for (ex, ey, ew, eh) in eyes:
                # Draw a rectangle around each eye
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Eye Tracking', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
