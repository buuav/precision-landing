import cv2
import numpy as np
def draw_keypoints(vis, keypoints, color = (0, 255, 255)):
    for kp in keypoints:
        x, y = kp.pt
        d = kp.size
        cv2.circle(vis, (int(x), int(y)), int(d), color)

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
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

        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 :
            self.detector = cv2.SimpleBlobDetector(params)
        else : 
            self.detector = cv2.SimpleBlobDetector_create(params)

        cam = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()

    
    def get_frame(self):
        success, frame = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        keypoints = self.detector.detect(frame)
        im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


        ret, jpeg = cv2.imencode('.jpg', im_with_keypoints)
        return jpeg.tostring()