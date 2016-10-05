import cv2
import sys

# Get user supplied values
imagePath = sys.argv[1]     #pass in the image and cascade names as command-line arguments
cascPath = sys.argv[2]

#create the haar cascade & init it with the face cascade,it loads the face cascade into memory  so its ready
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #convert it to grayscale for cv2

#detectmultiscale fnt gen fnt that detects objects,since we called it on facecascade,it detects faces
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)       #Since some faces may be closer to the camera, they would appear bigger than those faces in the back.
            # scale factor compensates for this.
        #The detection algorithm uses a moving window to detect objects. minNeighbors defines how many
            #  objects are detected near the current one before it declares the face found.
        #minSize,  gives the size of each window.

print "Found {0} faces!".format(len(faces))
    #The function returns a list of rectangles where it believes it found a face


for (x, y, w, h) in faces:  #loop over where it thinks it found something.
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #This function returns 4 values: the x and y location of the rectangle,
    #and the rectangles width & height

cv2.imshow("Faces found" ,image)        #values to draw a rectangle using the built-in rectangle() function.
cv2.waitKey(0)                          # wait for the user to press a key.

