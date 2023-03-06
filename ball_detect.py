#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('images/frame0050.jpg')
hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
low_orange = np.array([12,50,50])
high_orange = np.array([15,255,255])
orange_mask = cv2.inRange(hsv_frame, low_orange, high_orange)
orange = cv2.bitwise_and(img,img,mask=orange_mask)
gray = cv2.cvtColor(orange, cv2.COLOR_BGR2GRAY)

detected_circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,2, 300)

# Draw circles that are detected.
if detected_circles is not None:
    detected_circles = np.uint16(np.around(detected_circles))
    print("Coordinates:" + str(detected_circles[0, 0, 0]) + "," + str(detected_circles[0, 0, 1]))    


#----UNCOMMENT THIS BLOCK FOR IMAGE VISUALIZATION---#
    # for pt in detected_circles[0, :]:
    #     a, b, r = pt[0], pt[1], pt[2]

    #     # Draw the circumference of the circle.
    #     cv2.circle(img, (a, b), r, (0, 255, 0), 2)

    #     # Draw a small circle (of radius 1) to show the center.
    #     cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
    #     cv2.imshow("Detected Circle", img)
    #     cv2.waitKey(0)

else:
    print("No Circles")