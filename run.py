from cv2 import cv2
import sys

# Get user supplied values
imagePath = sys.argv[1]     #pass in the image and cascade names as command-line arguments
cascPath = sys.argv[2]

# Create the haar cascade & and initialize it with our face cascade.
# loads the face cascade into memory so it’s ready
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #convert it to grayscale for cv2

                    #detectMultiScale function is a general function that detects objects. Since we are calling it on the face cascade, that’s what it detects.
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)       #Since some faces may be closer to the camera, they would appear bigger than those faces in the back.
            # scale factor compensates for this.
        #The detection algorithm uses a moving window to detect objects. minNeighbors defines how many
            #  objects are detected near the current one before it declares the face found.
        #minSize,  gives the size of each window.

