from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

import numpy as np
from common import draw_str

def draw_keypoints(vis, keypoints, color = (0, 255, 255)):
    for kp in keypoints:
        x, y = kp.pt
        d = kp.size
        cv2.circle(vis, (int(x), int(y)), int(d), color)

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

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
font = cv2.FONT_HERSHEY_SIMPLEX

# allow the camera to warmup
time.sleep(0.1)
 
# capture frames from the camera
for cframe in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    frame = cframe.array
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    keypoints = detector.detect(frame)
    #im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    draw_keypoints(frame, keypoints)
    
    
    
    numkey = len(keypoints)
    for i in range(numkey):
        cv2.putText(frame, '%4.4f,%4.4f' % (keypoints[i].pt[0],keypoints[i].pt[1]), (20, 20*(i+1)), font, 0.5,(255,255,255),1)


    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
