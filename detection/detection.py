#!/usr/bin/env python

import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospy

class detect_manager:
    def __init__(self,):
        # The topic published by Intel435i camera.
        self.image_topic = "/device_0/sensor_1/Color_0/image/data"
        self.depth_topic = "/device_0/sensor_0/Depth_0/image/data"
        # Publish the detections results (image_raw).
        self.published_image_topic = "/detections_results"
        # Used for convert the msg to opencv image.
        self.bridge = CvBridge()
        # If receive image topic from camera, run detect() to detect the ball.
        self.image_sub = rospy.Subscriber(
            self.image_topic, Image, self.detect, queue_size=1, buff_size=2**24)
        self.depth_sub = rospy.Subscriber(
            self.depth_topic, Image, self.processDepth, queue_size=1, buff_size=2**24)
        self.pub_viz_ = rospy.Publisher(
            self.published_image_topic, Image, queue_size=10)
                
        # Define the publisher to publish the detection results (image_raw).
        self.pub_viz_ = rospy.Publisher(
            self.published_image_topic, Image, queue_size=10)
      
        rospy.spin()
        
    
    def processDepth(self, data):
        depth = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
    
    def detect(self, data):
        try:
            # Convert the msg (from 435i) to opencv bgr8 forrmat array.
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e) 
        img = self.cv_image


        small_to_large_image_size_ratio =0.5
        img = cv2.resize(img, # original image
                       (0,0), # set fx and fy, not the final size
                       fx=small_to_large_image_size_ratio, 
                       fy=small_to_large_image_size_ratio, 
                       interpolation=cv2.INTER_NEAREST)


        # Convert color space to HSV.
        hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Define the low and high range for the detection of ball.
        low_orange = np.array([10,140,20])
        high_orange = np.array([25,255,255])
        # The pixel lower than the low_orange or larger than the high_orange will be set to 0.
        orange_mask = cv2.inRange(hsv_frame, low_orange, high_orange)
        orange = cv2.bitwise_and(img,img,mask=orange_mask)
        # Get gray image of the detected ball.
        gray = cv2.cvtColor(orange, cv2.COLOR_BGR2GRAY)
        # get the circles of the detected ball.
        detected_circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,3, 300, 75, 45)
        
        # Draw circles that are detected.
        if detected_circles is not None:
            detected_circles = np.uint16(np.around(detected_circles))
            print("Coordinates:" + str(detected_circles[0, 0, 0]) + "," + str(detected_circles[0, 0, 1]))    


        # ----FOR IMAGE VISUALIZATION---#
            for pt in detected_circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]
                # Draw the circumference of the circle.
                cv2.circle(img, (a, b), r, (0, 255, 0), 2)

                # Draw a small circle (of radius 1) to show the center.
                cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
                pub_img = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
                self.pub_viz_.publish(pub_img)
                # cv2.imshow("Detected Circle", img)
                # cv2.waitKey(0)
                
        # Convert image to msg for publishing.
        img = self.bridge.cv2_to_imgmsg(img, encoding="passthrough")
        self.pub_viz_.publish(img)
     
        # else:
        #     print("No Circles")
    

    
    
    
if __name__ == '__main__':
    rospy.init_node("detection ball")
    rospy.loginfo("start detection node")
    dm = detect_manager()
