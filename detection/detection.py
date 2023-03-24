#!/usr/bin/env python

import numpy as np
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovariance, Point, Pose, PoseWithCovarianceStamped
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge, CvBridgeError
import rospy
import copy

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
        self.frame_id = "/pose_frame"
        self.init_Marker()
        rospy.spin()

    def init_Marker(self):
        self.ball = Marker()

        self.ball.header.frame_id = "/my_frame_ball"
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
        detected_circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 3, 300, 75, 45)

        # Draw circles that are detected.
        if detected_circles is not None:
            detected_circles = np.uint16(np.around(detected_circles))
            x_val = detected_circles[0, 0, 0]*1/small_to_large_image_size_ratio
            y_val = detected_circles[0, 0, 1]*1/small_to_large_image_size_ratio
            print("Coordinates:" + str(y_val) + "," + str(x_val))
            print("Depth:" + str(current_depth[int(y_val), int(x_val)]))
            depth_val = current_depth[int(y_val), int(x_val)]

            r_val, c_val = x_val, y_val
            theta_x = abs(c_val-320) * 74 // 320
            theta_y = abs(r_val-240) * 62 // 240
            print("(x y z): ({} {} {}) || (ThetaX ThetaY): ({} {})".format(c_val-320, r_val-240, depth_val, theta_x, theta_y))

            x_variance = 13.3333333
            y_variance = 10.0
            depth_variance = depth_val*0.02
            # TODO: Create a new updated belief using the means (measurements) and these covariances:
            #      These are arbitrarily picked for the x,y , but we can figure that out more next week
            #      so we can get a model going before this is due lol.

            # It will be a bit weird because the coordinate space our objec will be in is pixel,pixel,mm which is wrong
            # but workable for this assignment I think
            pose = Pose(
                position=Point(
                    x=x_val / 100,
                    y=y_val / 100,
                    z=depth_val / 500
                )
            )

            pose_stamped = PoseWithCovarianceStamped(
                Header(
                    seq=self.seq,
                    stamp=rospy.Time.now(),
                    frame_id=self.frame_id
                ),
                PoseWithCovariance(
                    pose=pose
                )
            )
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
            # TODO: Update the belief with a zero mean, MASSIVE covariance measurement to signal that the
            # ball is not in sight at all, ie x,y,z=0 x,y,z-cov = (inf/really big)

        # Convert image to msg for publishing.
        img = self.bridge.cv2_to_imgmsg(img, encoding="passthrough")
        self.pub_viz_.publish(img)


if __name__ == '__main__':
    rospy.init_node("detection ball")
    rospy.loginfo("start detection node")
    dm = detect_manager()
