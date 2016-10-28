import cv2
import numpy as np
import matplotlib.pylab as plt
from pprint import pprint

def order_corners(corners):
    cx, cy = np.average(corners, 0)
    top = []
    bot = []

    for i in range(4):
        if corners[i][1] < cy:
            top.append(corners[i])
        else:
            bot.append(corners[i])

    top = sorted(top, key = lambda pt:pt[0], reverse = False)
    bot = sorted(bot, key = lambda pt:pt[0], reverse = True)
    return top + bot

def get_intersection(line1, line2):
    pt1 = line1[0]
    pt2 = line1[1]
    pt3 = line2[0]
    pt4 = line2[1]

    (x1, y1) = (pt1[0], pt1[1])
    (x2, y2) = (pt2[0], pt2[1])
    (x3, y3) = (pt3[0], pt3[1])
    (x4, y4) = (pt4[0], pt4[1])

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None
    else:
        x_numer = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
        y_numer = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
        return (x_numer/denom, y_numer/denom)

def test():
    img = cv2.imread('snap.jpg',1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #stylized - assumption based

    # lower mask (0-10)houghlines
    lower_red = np.array([0,127,50])
    upper_red = np.array([17,255,255])
    mask0 = cv2.inRange(hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([175,127,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_green = np.array([45,75,50])
    upper_green = np.array([75,255,255])
    mask2 = cv2.inRange(hsv, lower_green, upper_green)
    mask = mask0 + mask1 + mask2

    kernel = np.ones((5,5), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations = 1)
    rehydrated = cv2.dilate(mask, kernel, iterations = 1)

    (cnts, hier) = cv2.findContours(rehydrated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    quads = []
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            quads.append(approx)

    quad = max(quads, key = lambda x: cv2.contourArea(x))

    points = np.squeeze(quad)
    pprint(points)
    corners = order_corners(points)
    pprint(corners)
    diag1 = (corners[0],corners[2])
    diag2 = (corners[1],corners[3])
    center = get_intersection(diag1, diag2)


    out = img.copy()
    cv2.drawContours(out, quads, -1, (0,255,0), 3)
    cv2.line(out, (corners[0][0],corners[0][1]), (corners[2][0],corners[2][1]), (0,255,255), thickness=2, lineType=8, shift=0)
    cv2.line(out, (corners[1][0],corners[1][1]), (corners[3][0],corners[3][1]), (0,255,255), thickness=2, lineType=8, shift=0)
    cv2.circle(out, (int(center[0]),int(center[1])), 10, (0,0,0), 3, lineType = cv2.LINE_AA, shift = 0)
    cv2.imshow('Output', mask)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test()
