#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      henghuiz
#
# Created:     04/10/2015
# Copyright:   (c) henghuiz 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import cv2
from common import draw_str

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
 
# Change thresholds
params.minThreshold = 10;
params.maxThreshold = 200;
 
# Filter by Area.
params.filterByArea = True
params.maxArea = 1500
 
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1
 
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.9
 
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01
 
# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
    detector = cv2.SimpleBlobDetector(params)
else : 
    detector = cv2.SimpleBlobDetector_create(params)

cap = cv2.VideoCapture('test_video.3gp')

while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    keypoints = detector.detect(frame)
    im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    
    
    
    numkey = len(keypoints)
    for i in range(numkey):
        draw_str(im_with_keypoints, (20, 20*(i+1)), '%4.4f,%4.4f' % (keypoints[i].pt[0],keypoints[i].pt[1]))


    cv2.imshow('frame',im_with_keypoints)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()