from cv2 import cv2
import sys

# Get user supplied values
imagePath = sys.argv[1]     #pass in the image and cascade names as command-line arguments
cascPath = sys.argv[2]

# Create the haar cascade & and initialize it with our face cascade.
faceCascade = cv2.CascadeClassifier(cascPath)