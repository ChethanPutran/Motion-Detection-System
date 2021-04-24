# Importing neccessary modules
import cv2
import numpy as np

# Loading face cascades for face detection
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Loading image
image = cv2.imread("news.jpg")
testImage = np.copy(image)

# Converting the colored image to gray scale image
grayScaleImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)

#Detecting the face and it returns the dimensions of the face(x-start,y-start,width,height)
faces = faceCascade.detectMultiScale(
    grayScaleImage, scaleFactor=1.1, minNeighbors=5)

#print(faces)

#Getting the dimensions of the face
for x,y,w,h in faces:
    
    #Drawing the rectangle around the face
    cv2.rectangle(testImage,(x,y),(x+w,y+h),(0,255,0),3)

#Resizing the image
resizedImg = cv2.resize(testImage,(int(testImage.shape[1]/3),int(testImage.shape[0]/3)))

# Dispalying the image
cv2.imshow("Image", resizedImg)

# Waiting for key to be pressed
cv2.waitKey(0)

# Destroying the window
cv2.destroyAllWindows()
