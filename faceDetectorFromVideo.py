# Importing neccessary modules
import cv2
import numpy as np

# Loading face cascades for face detection
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Caputring video
video = cv2.VideoCapture(0)

while True:
    
    check, frame = video.read()

    # Converting the colored image/frame to gray scale image
    grayScaleImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detecting the face and it returns the dimensions of the face(x-start,y-start,width,height)
    faces = faceCascade.detectMultiScale(
        grayScaleImage, scaleFactor=1.1, minNeighbors=5)

    #print(faces)

    #Getting the dimensions of the face
    for x,y,w,h in faces:
        
        #Drawing the rectangle around the face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

    #Resizing the image
    resizedImg = cv2.resize(frame,(int(frame.shape[1]/2),int(frame.shape[0]/2)))

    # Dispalying the image
    cv2.imshow("Image", resizedImg)

    # Waiting for key to be pressed
    if cv2.waitKey(1) == ord('q'):
        break
    continue

video.release()

# Destroying the window
cv2.destroyAllWindows()
