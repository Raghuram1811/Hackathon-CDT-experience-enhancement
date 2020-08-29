#Author : Raghuram GVS

import cv2
import imutils
import numpy as np
import math
import pyscreenshot

#capture webcam
camera = cv2.VideoCapture(0)

#load images of same pixel dimensions
# Pixel dimensions s_img.shape[0] = 103, s_img.shape[1] = 188
s_img1 = cv2.imread("RosheOne_Quarter300px.png", -1) 
s_img2 = cv2.imread("rosheone_quarter_red0719.png",-1) 
s_img3 = cv2.imread("rosheone_quarter_blue0719.png",-1)
s_img4 = cv2.imread("rosheone_quarter_floral0719.png", -1)
    
#set upper and lower bounds for green detection
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

#intialize image selector count variable to 0
count = 0

#this loop continuously grabs the frame from the webcam and checks for two green contours
#once found, the middle of the contours is calculated, and the image is overlayed
while True:

    #grab frame from webcam
    grabbed, frame = camera.read()

    #flip frame to make it look like mirror
    frame = cv2.flip(frame,1)

    #resize frame
    frame = imutils.resize(frame, width=800,height=600)

    s_img = None

    #select which image will be overlayed
    if count == 0:
        s_img = s_img1
    elif count == 1:
        s_img = s_img2
    elif count == 2:
        s_img = s_img3
    elif count == 3:
        s_img = s_img4

    # Create the mask for the image
    orig_mask = s_img[:,:,3]
 
    # Create the inverted mask for the image
    orig_mask_inv = cv2.bitwise_not(orig_mask)
 
    # Convert image to BGR
    # and save the original image size (used later when re-sizing the image)
    s_img = s_img[:,:,0:3]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #create mask
    mask = cv2.inRange(hsv, greenLower, greenUpper)

    #clean up mask
    mask = cv2.erode(mask, None, iterations=2)   #Removes any additional pixels that are part of the shoe image
    mask = cv2.dilate(mask, None, iterations=2)  #Adds additional thickness to the existing image

    #findContours in mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(cnts) > 1:

        #Enclose the green instances with circle and take measurements
        d = max(cnts, key=cv2.contourArea)
        index = cnts.index(d)
        cnts.remove(d)
        c = max(cnts, key=cv2.contourArea)

        ((x,y), radius) = cv2.minEnclosingCircle(c)
        ((a,b), r) = cv2.minEnclosingCircle(d)

        #avoid negative dimensions
        
        if x <= a:
            xDist = a - x
            xCoord = int(x + (xDist / 2))
        else:
            xDist = x - a
            xCoord = int(a + (xDist / 2))

        if y <= b:
            yDist = b - y
            yCoord = int(y + (yDist / 2))
        else:
            yDist = y - b
            yCoord = int(b + (yDist / 2))

        #dynamic scaling
        hyp = math.sqrt((xDist * xDist) + (yDist * yDist));
        img_hyp = math.sqrt(s_img.shape[0]*s_img.shape[0] + s_img.shape[1]*s_img.shape[1])
        scaleFactor = (hyp / img_hyp) * 1.2

        #create 'scaled' image dimensions    
	snkrWidth = int(s_img.shape[1] * scaleFactor)
	snkrHeight = int(s_img.shape[0] * scaleFactor) 
	halfWidth = int(snkrWidth/2)
	halfHeight = int(snkrHeight/2)

        y_roiStart = yCoord-halfHeight
        y_roiEnd = yCoord+halfHeight
        x_roiStart = xCoord-halfWidth 
        x_roiEnd = xCoord+halfWidth

        #adjust ends as necessary based on window size
	if y_roiStart < 0:
            y_roiStart = 0
            
        if y_roiEnd > 450:
            y_roiEnd = 450
            
        if x_roiStart < 0:
            x_roiStart = 0

        if x_roiEnd > 800:
            x_roiEnd = 800
            
	roi_height = y_roiEnd-y_roiStart
        roi_width = x_roiEnd-x_roiStart

        #grab region of interest centered around the middle of the two green instances
	roi_color = frame[y_roiStart:y_roiEnd, x_roiStart:x_roiEnd]

	#resize masks based on scaled dimensions
        snkr_img = cv2.resize(s_img, (roi_width,roi_height), interpolation = cv2.INTER_AREA)
        mask2 = cv2.resize(orig_mask, (roi_width,roi_height), interpolation = cv2.INTER_AREA)
        mask_inv2 = cv2.resize(orig_mask_inv, (roi_width,roi_height), interpolation = cv2.INTER_AREA)

        #create the background and foreground and recompile to create overlay image with
        #transparent alpha layer
	roi_bg = cv2.bitwise_and(roi_color,roi_color, mask=mask_inv2)
	roi_fg = cv2.bitwise_and(snkr_img, snkr_img, mask=mask2)
	snkr = cv2.add(roi_bg,roi_fg)

        #display more windows for debug
	#cv2.imshow("roi_bg",roi_bg)
	#cv2.imshow("roi_fg",roi_fg)
        
	#overlay the image over the frame
        frame[y_roiStart:y_roiEnd, x_roiStart:x_roiEnd] = snkr
        
    #display the frame and move the window towards the middle of the screen
    #the moveWindow positions may have to be adjusted based on individual window resolution
    cv2.imshow("Frame", frame)
    cv2.moveWindow("Frame", 200, 100)

    #if the 's' key is pressed, take a screenshot and display it
    #the bbox variable may have to be adjusted based on individual window resolution
    if cv2.waitKey(1) & 0xFF == ord("s"):
        screenshot = pyscreenshot.grab(bbox=(200,100,1000,580))
        screenshot.show()

    # if the 'z' key is pressed, cycle through images (low fidelity because of loop structure)
    if cv2.waitKey(1) & 0xFF == ord("z"):
        if count == 3:
            count = 0
        else:
            count = count + 1
        
    # if the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
 
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
