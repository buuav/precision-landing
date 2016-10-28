import sides
import cv2
import numpy as np
from pprint import pprint

cap = cv2.VideoCapture(0)
while(1):
    _,frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0,50,70])
    upper_red = np.array([5,255,255])
    mask0 = cv2.inRange(hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([175,50,70])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)


    lower_green = np.array([45,50,50])
    upper_green = np.array([80,255,255])
    mask2 = cv2.inRange(hsv, lower_green, upper_green)
    mask = mask0 + mask1 + mask2

    small_kernel = np.ones((1,1), np.uint8)
    regular_kernel = np.ones((5,5), np.uint8)
    #large_kernel = np.ones((10,10), np.uint8)

    noise_filter = cv2.erode(mask, regular_kernel, iterations = 1)
    noise_filter2 = cv2.dilate(noise_filter, regular_kernel, iterations = 3)
    cv2.imshow('big',mask)
    noise_filter = cv2.erode(noise_filter2, small_kernel, iterations = 10)


    (cnts, hier) = cv2.findContours(noise_filter.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    quads = []
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)
        if len(approx) == 4:
            quads.append(approx)

    out = frame.copy()
    if len(quads) > 0:
        quad = max(quads, key = lambda x: cv2.contourArea(x))
        if (cv2.contourArea(quad) > 100):
            points = np.squeeze(quad)
            pprint(points)
            corners = sides.order_corners(points)
            pprint(corners)
            diag1 = (corners[0],corners[2])
            diag2 = (corners[1],corners[3])
            center = sides.get_intersection(diag1, diag2)
            if center:
                cv2.drawContours(out, quads, -1, (0,255,0), 3)
                cv2.line(out, (corners[0][0],corners[0][1]), (corners[2][0],corners[2][1]), (0,255,255), thickness=2, lineType=8, shift=0)
                cv2.line(out, (corners[1][0],corners[1][1]), (corners[3][0],corners[3][1]), (0,255,255), thickness=2, lineType=8, shift=0)
                cv2.circle(out, (int(center[0]),int(center[1])), 10, (0,0,0), 3, lineType = cv2.LINE_AA, shift = 0)
    cv2.imshow('Mask', noise_filter)
    cv2.imshow('Output', out)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
