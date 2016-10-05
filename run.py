import cv2
import sys

cascPath = sys.argv[1]  # pass in cascade name as command-line arguments
faceCascade = cv2.CascadeClassifier(
    cascPath)  # create the haar cascade & init it with the face cascade,it loads the face cascade into memory

video_capture = cv2.VideoCapture(0)  # sets video source to webcam
video_capture.set(3,640)
video_capture.set(4,480)
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    # convert it to grayscale for cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detectmultiscale fnt gen fnt that detects objects,since we called it on facecascade,it detects faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    # Since some faces may be closer to the camera, they would appear bigger than those faces in the back.
    # scale factor compensates for this.
    # The detection algorithm uses a moving window to detect objects. minNeighbors defines how many
    #  objects are detected near the current one before it declares the face found.
    # minSize,  gives the size of each window.
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # This function returns the x and y location of the rectangle,
        # and the rectangles width & height

        # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # exit when button q is pressed

# releasing the capture
video_capture.release()
cv2.destroyAllWindows()
