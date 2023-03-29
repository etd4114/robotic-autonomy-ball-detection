#!/usr/bin/env python

import numpy as np
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovariance, Point, Pose, PoseWithCovarianceStamped
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge, CvBridgeError
import rospy, math
import copy, time

from scipy import signal
import imutils

class detect_manager:
    def __init__(self,):
        # The topic published by Intel435i camera.
        self.image_topic = "/device_0/sensor_1/Color_0/image/data"
        self.depth_topic = "/device_0/sensor_0/Depth_0/image/data"
        # Publish the detections results (image_raw).
        self.published_image_topic = "/detections_results"
        self.published_pose_covariance_topic = "/pose_covariance"
        self.published_ball_visualize_topic = "/ball_visualize"
        # Used for convert the msg to opencv image.
        self.bridge = CvBridge()
        # If receive image topic from camera, run detect() to detect the ball.
        self.image_sub = rospy.Subscriber(
            self.image_topic, Image, self.detect, queue_size=1, buff_size=2**24)
        self.depth_sub = rospy.Subscriber(
            self.depth_topic, Image, self.processDepth, queue_size=1, buff_size=2**24)

        # Define the publisher to publish the detection results (image_raw).
        self.pub_viz_ = rospy.Publisher(
            self.published_image_topic, Image, queue_size=10)
        self.pub_pose_covariance = rospy.Publisher(
            self.published_pose_covariance_topic, PoseWithCovarianceStamped, queue_size=10)
        self.pub_visualize_ball = rospy.Publisher(
            self.published_ball_visualize_topic, Marker, queue_size=10)
        self.depth = None
        self.seq = 0
        self.frame_id = "/camera_frame"

        ## temporal smooth for ball detection
        self.ball_detect_mode = 'Contour'   # 'Hough
        self.ball_depth_qsize = 10
        self.ball_depth_queue = []


        self.init_Marker()
        rospy.spin()

    def init_Marker(self):
        self.ball = Marker()

        self.ball.header.frame_id = self.frame_id
        self.ball.header.stamp = rospy.Time.now()
        self.ball.ns = "basic_shapes"
        self.ball.id = self.seq
        self.ball.type = Marker.SPHERE
        self.ball.action = Marker.MODIFY
        self.ball.pose.orientation.x = 0.0
        self.ball.pose.orientation.y = 0.0
        self.ball.pose.orientation.z = 0.0
        self.ball.pose.orientation.w = 1.0

        self.ball.scale.x = 1.0
        self.ball.scale.y = 1.0
        self.ball.scale.z = 1.0
        self.ball.color.r = 1.0
        self.ball.color.g = 0.7
        self.ball.color.b = 0.0
        self.ball.color.a = 1.0
        self.ball.lifetime = rospy.Time(0, 10)

    def processDepth(self, data):
        self.depth = self.bridge.imgmsg_to_cv2(
            data, desired_encoding="passthrough")

    def detect(self, data):
        self.seq += 1

        current_depth = copy.deepcopy(self.depth)

        try:
            # Convert the msg (from 435i) to opencv bgr8 forrmat array.
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        img = self.cv_image

        small_to_large_image_size_ratio = 0.5
        img = cv2.resize(img,  # original image
                         (0, 0),  # set fx and fy, not the final size
                         fx=small_to_large_image_size_ratio,
                         fy=small_to_large_image_size_ratio,
                         interpolation=cv2.INTER_NEAREST)

        # Convert color space to HSV.
        hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Define the low and high range for the detection of ball.
        low_orange = np.array([10, 140, 20])
        high_orange = np.array([25, 255, 255])
        # The pixel lower than the low_orange or larger than the high_orange will be set to 0.
        orange_mask = cv2.inRange(hsv_frame, low_orange, high_orange)
        orange = cv2.bitwise_and(img, img, mask=orange_mask)
        # Get gray image of the detected ball.
        gray = cv2.cvtColor(orange, cv2.COLOR_BGR2GRAY)
        # get the circles of the detected ball.

        detected_circles = None
        if self.ball_detect_mode == 'Hough':
            detected_circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, 3, 300, 75, 45)
        else:
            orange_mask = cv2.erode(orange_mask, None, iterations=2)
            orange_mask = cv2.dilate(orange_mask, None, iterations=2)
            cnts = cv2.findContours(orange_mask.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            center = None

            # only proceed if at least one contour was found
            if len(cnts) > 0:
                # find the largest contour in the mask, then use
                # it to compute the minimum enclosing circle and
                # centroid
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                if radius > 10:
                    detected_circles = [[[x, y, radius]]]
        
        # Draw circles that are detected.
        if detected_circles is not None:

            detected_circles = np.uint16(np.around(detected_circles))
            x_val = detected_circles[0, 0, 0]*1/small_to_large_image_size_ratio
            y_val = detected_circles[0, 0, 1]*1/small_to_large_image_size_ratio
            depth_val = current_depth[int(y_val), int(x_val)]

            depth_adjust = 250
            depth_val = max(0, depth_val - depth_adjust)

            # ## Spatial smooth   (average filter with valid (>0) data)
            # fsize = 5
            # w = fsize // 2
            # r1, r2, c1, c2 = max(int(y_val)-w, 0), min(int(y_val)+w+1, 479), max(int(x_val)-w, 0), min(int(x_val)+w+1, 639)
            # #print(current_depth[r1:r2, c1:c2])
            # depth_boxsum = np.sum(current_depth[r1:r2, c1:c2])
            # depth_validcnt = np.count_nonzero(current_depth[r1:r2, c1:c2])
            # depth_val = depth_boxsum // max(depth_validcnt, 1)
            # ## ===== ##

            ## Temporal Smooth of depth value
            if depth_val == 0:  ## Bad pixel, use temporal data
                depth_val = np.average(np.array(self.ball_depth_queue))
            else:   ## Put new data in the queue
                if len(self.ball_depth_queue) > 0 and depth_val > self.ball_depth_queue[-1] * 1.2:   ## In case the depth difference between 2 frame is too big
                    depth_val = self.ball_depth_queue[-1] * 1.2
                if len(self.ball_depth_queue) == self.ball_depth_qsize:
                    self.ball_depth_queue = self.ball_depth_queue[1:]
                self.ball_depth_queue.append(depth_val)
            depth_val = np.average(np.array(self.ball_depth_queue))
            depth_val = int(depth_val)
            #print(self.ball_depth_queue, 'depth_val:', depth_val)
            ## ===== ##

            if depth_val < 175: ## minimum detectable region according to the manual
                print("No Circles (Invalid depth: {})".format(depth_val))
                img = self.bridge.cv2_to_imgmsg(img, encoding="passthrough")
                self.pub_viz_.publish(img)
                return
            
            depth_scaling_factor = 100.0
            depth_val /= depth_scaling_factor


            ## Calculate the angle from the pixel
            ## Use the center of the picture as the origin of the coordinate
            w_val, h_val = x_val, y_val
            x_fixed, y_fixed = (w_val-320), (240-h_val)
            theta_x = x_fixed * 74 // 640
            theta_y = y_fixed * 62 // 480
            # print("(x y z): ({} {} {}) || (ThetaX ThetaY): ({} {})".format(x_fixed, y_fixed, depth_val, theta_x, theta_y))
            x_coordinate, y_coordinate = depth_val*math.sin(math.radians(theta_x)),depth_val*math.cos(math.radians(theta_x))
            #print("x,y = {},{}, D: {}(cm), THETA: {}".format(x_coordinate, y_coordinate, depth_val, theta_x))
            print("x,y = {},{}, D: {}(cm), THETA: {}".format(int(x_coordinate*10), int(y_coordinate)*10, int(depth_val*10), theta_x))

            ## Covariance matrix setting up:
            ## For the depth sensor we can estimate a variance of about (2% of d)^2
            ## For the camera angle we can estimate a variance of about 1.5 degrees
            ## So with that we can setup our matrix as follows:
            ##
            ##  [((0.02*d)^2)sin(radians(theta))         0]
            ##  [0         ((0.02*d)^2)cos(radians(1.5))]
            ##

            ## The way that the covariance matrix is setup within the message is that it is a 36 length array
            ## where the diagonal are the variances so index 0 is variance of x and index 7 is variance of y and so on...
            ## I'll just put the whole matrix here:
            
            #measured error for the depth sensor was about 0.05025 * distance
            r_cov = 0.0525*depth_val
            t_cov = math.radians(.55)
            # y_cov = r_cov * r_cov * math.cos(t_cov) * math.cos(t_cov)
            # x_cov = r_cov * r_cov * math.sin(t_cov) * math.sin(t_cov)

            theta = math.radians(theta_x)

            R = np.array([[math.sin(theta), depth_val*math.cos(theta)], [math.cos(theta), -depth_val*math.sin(theta)]])
            Pol = np.array([[r_cov*r_cov, 0], [0, t_cov*t_cov]])
            cov_mat = np.matmul(np.matmul(R,Pol), R.T)

            covariance = [cov_mat[0,0],  cov_mat[0,1],    0.0,    0.0,    0.0,    0.0,
                          cov_mat[1,0],   cov_mat[1,1],   0.0,    0.0,    0.0,    0.0,
                          0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
                          0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
                          0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
                          0.0,    0.0,    0.0,    0.0,    0.0,    0.0]
            
            pose = Pose(
                position=Point(
                    x= x_coordinate,
                    y= y_coordinate,
                    z= 0
                )
            )

            pose_stamped = PoseWithCovarianceStamped(
                Header(
                    seq=self.seq,
                    stamp=rospy.Time.now(),
                    frame_id=self.frame_id
                ),
                PoseWithCovariance(
                    pose=pose,
                    covariance=covariance
                ))
            self.pub_pose_covariance.publish(pose_stamped)
            self.ball.pose = pose
            self.pub_visualize_ball.publish(self.ball)

        # ----FOR IMAGE VISUALIZATION---#
            for pt in detected_circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]
                # Draw the circumference of the circle.
                cv2.circle(img, (a, b), r, (0, 255, 0), 2)
                # Draw a small circle (of radius 1) to show the center.
                cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
        else:
            print("No Circles")

        # Convert image to msg for publishing.
        img = self.bridge.cv2_to_imgmsg(img, encoding="passthrough")
        self.pub_viz_.publish(img)


if __name__ == '__main__':
    rospy.init_node("detection ball")
    rospy.loginfo("start detection node")
    dm = detect_manager()
