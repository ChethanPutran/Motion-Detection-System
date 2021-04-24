
#Program Motion Detector -by Chethan


# Importing neccessary modules
import cv2
import numpy as np
from datetime import datetime
import pandas
import os
from plotter import Plot


firstFrame = None
contourArea = 8000
fileName = "motion.csv"


def drawRectangle(frame, contours):
    global movement
    
    for contour in contours:

        if cv2.contourArea(contour) < contourArea:
           
            continue
        
        movement = 1
        # Getting the dimensions of the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # Drawing rectangle around the contour having thickness 3px
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
    
    return frame, movement


def cannyImage(image):
    # 1.Conveting coloured image to grayscale image
    grayScaleImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 2.Reducing noise and smoothening image
    bluredGSImage = cv2.GaussianBlur(grayScaleImage, (21, 21), 0)

    # #Determing the edges based on gradient(taking derivative(f(x,y)) x->along width of the message y-> along height)
    # canny = cv2.Canny(bluredGSImage,50,150)#(image,low_threshold,hight_threshold)

    return bluredGSImage


movements = [None,None]
motionTimes = []
df = pandas.DataFrame(columns=["Entered Time","Left Time"])

# Caputring video
video = cv2.VideoCapture(0)

while True:
    # Reading each frame from the video
    check, frame = video.read()

    # Initializing movement to 0
    movement = 0
   
    # Getting the first frame or image of the video
    if firstFrame is None:
        firstFrame = cannyImage(frame)
        continue

    currentFrame = cannyImage(frame)

    # Getting difference between first frame and current frame
    deltaFrame = cv2.absdiff(firstFrame, currentFrame)

    # Getting threshold frame
    thresholdFrame = cv2.threshold(deltaFrame, 30, 255, cv2.THRESH_BINARY)[1]

    # Removing noise
    thresholdFrame = cv2.dilate(thresholdFrame, None, iterations=4)

    # Getting contours
    (contours, _) = cv2.findContours(thresholdFrame.copy(),
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Drawing rectangle arround the contour if it has desired area
    frameWithRectangle, movement = drawRectangle(frame, contours)
    
    movements.append(movement)
    movements = movements[-3:]
    
    if movements[-1] == 1 and movements[-2] == 0 :
        motionTimes.append(datetime.now())

    if movements[-1] == 0 and movements[-2] == 1 :
        motionTimes.append(datetime.now())
    
    # Dispalying the image
    cv2.imshow("Real", frameWithRectangle)
    # cv2.imshow("Image", deltaFrame)
    # cv2.imshow("Threshold", thresholdFrame)

    # Waiting for key to be pressed
    if cv2.waitKey(1) == ord('q'):
        if(movements[-1] == 1):
            motionTimes.append(datetime.now())
            
        break
    
    
try:
    for i in range(0,len( motionTimes),2):
        df = df.append({"Entered Time": motionTimes[i],"Left Time": motionTimes[i+1]},ignore_index=True)

    #Checking if file exits or not,if the file exits then the index is continued from privious results
    if (os.path.isfile(fileName)):
        
        df.to_csv("temp.csv")
        
        with open(fileName,'r') as destination:
            
            lines = destination.readlines()
            lastLine = lines[-1]
            
        lastIndex = lastLine.split(',')[0]  
        
        with open(fileName,'a') as destination: 
            
            with open("temp.csv",'r') as source:
                
                lines = source.readlines()
                
                for i in range(1,len(lines)):
                    items = lines[i].split(',')
                    items[0] = int(lastIndex)+ (i+1)
                    line = ','.join(map(str,items)) 
                    destination.write(line)
            
        os.remove("temp.csv")               
    else:
        df.to_csv(fileName)
except Exception as e:
    print(e)
    print("Could not create file")          

video.release()

# Destroying the window
cv2.destroyAllWindows()

plot = Plot(fileName)
plot.makePlot()
