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

