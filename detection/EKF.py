#!/usr/bin/env python


import numpy as np
# import human_walk
import matplotlib.pyplot as plt
import pdb
import rospy
from geometry_msgs.msg import PoseWithCovariance, Point, Pose, PoseWithCovarianceStamped, Quaternion
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
import math
import numpy as np
from utils import quaternion2euler, euler2quaternion
from visualization_msgs.msg import Marker

class EKF():
    def __init__(self, nk, dt, X, U):
        self.nk = nk
        self.dt = 1
        self.X = X
        self.U = U
        self.measurement_sub_topic = "/pose_covariance"
        self.measurement_pub_topic = "/pose_corrected"
        self.published_ball_visualize_topic = "/ball_visualize_corrected"

        # <--------<< Initialize corection covariance
        self.Sigma_init = np.array([[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]])
        # <--------<< Should be updated with variance from the measurement
        self.Q = np.array([[1, 0.0], [0.0, 1]])
        self.C = np.array([[1, 0, 0], [0, 1, 0]])
        # <--------<< Initialize Kalman Gain
        self.KalGain = np.random.rand(3, 2)
        self.initialized = False

        # <--------<< Subscribe to the ball pose topic
        self.measurement_sub = rospy.Subscriber(self.measurement_sub_topic, PoseWithCovarianceStamped, self.measurement_cb)
        self.measurement_pub = rospy.Publisher(self.measurement_pub_topic, PoseWithCovarianceStamped, queue_size=10)
        self.pub_visualize_ball = rospy.Publisher(self.published_ball_visualize_topic, Marker, queue_size=10)
        self.dl = 0.1
        self.Sx_k_k = self.Sigma_init
        self.mxkmGkm = np.array([0, 0, 0])
        self.Swkm = np.array([[1, 0], [0, 1]])
        self.observation = np.array([0,0])
        self.thetaXY = 0
        self.receieved_pose_with_covariance = PoseWithCovarianceStamped()
        self.seq = 0
        self.frame_id = "/camera_frame"
        self.init_Marker()
        
    def measurement_cb(self, data):
        self.receieved_pose_with_covariance = data
        self.observation = np.array([data.pose.pose.position.x, data.pose.pose.position.y])
        self.Swkm = np.array([[0.01, 0], [0, 0.01]])
        print("OBS" + str(self.observation))
        if not self.initialized:
            self.X = np.array([data.pose.pose.position.x, data.pose.pose.position.y, 0.0])
            self.initialized = True
        self.update()

    def prediction(self, xkm, U, Sigma_km1_km1):
        A, B = self.getGrad(xkm,U)
        x_new = np.matmul(A,xkm) + np.matmul(B,U)
        sigma_u = np.matmul(B ,np.matmul(self.Q, B.T))
        SxkGkm = np.matmul(A, np.matmul(Sigma_km1_km1, A.T)) + sigma_u
        return x_new, SxkGkm

    def correction(self, x_predict, Sx_k_km1, z_k, KalGain):
        # return x_predict, Sx_k_km1
        print("INNOVATION:" + str((z_k - np.matmul(self.C, x_predict))))
        x_new = x_predict + np.matmul(KalGain, (z_k - np.matmul(self.C, x_predict)))
        sigma_new = np.matmul((np.identity(3) - np.matmul(KalGain, self.C)), Sx_k_km1)
        return x_new, sigma_new

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
        self.ball.color.r = 0.0
        self.ball.color.g = 1.0
        self.ball.color.b = 0.0
        self.ball.color.a = 1.0
        self.ball.lifetime = rospy.Time(0, 10)

    def update(self):
        self.X_pred = self.X
        X_predicted, Sx_k_km1 = self.prediction(self.X, self.U, self.Sx_k_k)                        # PREDICTION STEP
        self.gainUpdate(Sx_k_km1) # GAIN UPDATE
        X_corrected, self.Sx_k_k = self.correction(X_predicted, Sx_k_km1, self.observation, self.KalGain)   # CORRECTION STEP
        

        # self.X.append(X_corrected)
        print("X k-1: {}".format(self.X))
        print("X_predicted: {}".format(X_predicted))
        # print("X_pred_cov: {}".format(Sx_k_km1))
        print("X_corrected: {}".format(X_corrected))
        # print("X_corrected_cov: {}".format(self.Sx_k_k))

        self.X = X_corrected
        self.X_pred = np.reshape(self.X_pred, [3, 1])
        self.X_corrected = np.reshape(
            X_corrected, [3, 1])   # <--------<< Publish

        quat = euler2quaternion([0,self.X[2],0])
        self.receieved_pose_with_covariance.pose.pose.position.x = self.X[0]
        self.receieved_pose_with_covariance.pose.pose.position.y = self.X[1]
        self.receieved_pose_with_covariance.pose.pose.orientation.x = quat[0]
        self.receieved_pose_with_covariance.pose.pose.orientation.y = quat[1]
        self.receieved_pose_with_covariance.pose.pose.orientation.z = quat[2]
        self.receieved_pose_with_covariance.pose.pose.orientation.w = quat[3]
        self.measurement_pub.publish(self.receieved_pose_with_covariance)
        self.ball.pose = self.receieved_pose_with_covariance.pose
        self.pub_visualize_ball.publish(self.ball)


    def gainUpdate(self, Sx_k_km1):
        print("SENS_COV: " + str(self.Swkm))
        print("X_COV: " + str(Sx_k_km1))
        self.KalGain = np.matmul(np.matmul(Sx_k_km1, self.C.T), (np.matmul(self.C, np.matmul(Sx_k_km1, self.C.T))) + self.Swkm)
        print("KAL: " + str(self.KalGain))
        
        return

    def dotX(self, x, U):
        # differential motion model
        l = 0.2
        r_wheel = 0.1
        wli = U[0]
        wri = U[1]
        thk = U[2]

        dxk = x[0] + (r_wheel / 2 * (wri + wli) * math.cos(x[2]))
        dyk = x[1] + (r_wheel / 2 * (wri + wli) * math.sin(x[2]))
        dthk = x[2] + (r_wheel / l * (wri - wli))
        return np.array([dxk, dyk, dthk])

    def getGrad(self, x, U):
        R = 0.1
        L = 0.2

        A = np.array([[1, 0, -R/2 * (U[0] + U[1]) * math.sin(x[2])], 
                      [0, 1,  R/2 * (U[0] + U[1]) * math.cos(x[2])], 
                      [0, 0,  1]])
        B = np.array([[R/2 * math.cos(x[2]), R/2 * math.cos(x[2])], 
                      [R/2 * math.sin(x[2]), R/2 * math.sin(x[2])], 
                      [R/L, -R/L]])
        return A, B
        


if __name__ == '__main__':
    rospy.init_node("EKF")

    # ---------------Define initial conditions --------------- #
    nk = 1       # <------<< Look ahead duration in seconds
    dt = 0.1       # <------<< Sampling duration of discrete model
    X = np.array([0.001, 0.001, 0.0])       # <------<< Initial State of the Ball
    U = np.array([0.1, 0.1])       # <------<< Initial input to the motion model

    filter = EKF(nk, dt, X, U)
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        # filter.update()
        rate.sleep()
