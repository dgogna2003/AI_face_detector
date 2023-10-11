import cv2
from random import randrange

# Load pre-trained data on face frontal from opencv [ Haar cascade algorithm ]
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
# img = cv2.imread('img3.PNG')

# Capture video from  default webcam
webcam = cv2.VideoCapture(0)

# Iterate forever loops over all the frames until webcam is closed
while True:
    # Read the current frame
    # Get the image out from the video
    successful_frame_read, frame = webcam.read()

    # Convert image to grey scale
    greyScaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces by returning coordinates of the rectangle surrounding the face
    face_coordinates = trained_face_data.detectMultiScale(greyScaled_img)

    # Draw rectangle around the face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, randrange(256), 0), 5)

    print(face_coordinates)  # Lists array of rectangle coordinates for each frame

    cv2.imshow('Face Detector Program', frame)
    key = cv2.waitKey(1)  # Display

    # Stop program if q or Q is pressed
    if key == 81 or key == 113:
        break


print("Code Works!!")
