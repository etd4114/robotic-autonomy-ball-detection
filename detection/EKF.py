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

        self.Q = np.array([[0.5, 0.0], [0.0, 0.5]])
        self.C = np.array([[1, 0, 0], [0, 1, 0]])
        self.Sx_k_k = np.array([[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]])
        self.Swkm = np.array([[1, 0], [0, 1]])
        self.observation = np.array([0,0])
        self.KalGain = np.random.rand(3, 2)
        self.initialized = False

        self.measurement_sub = rospy.Subscriber(self.measurement_sub_topic, PoseWithCovarianceStamped, self.measurement_cb)
        self.measurement_pub = rospy.Publisher(self.measurement_pub_topic, PoseWithCovarianceStamped, queue_size=10)
        self.pub_visualize_ball = rospy.Publisher(self.published_ball_visualize_topic, Marker, queue_size=10)

        self.receieved_pose_with_covariance = PoseWithCovarianceStamped()
        self.frame_id = "/camera_frame"
        
    def measurement_cb(self, data):
        self.receieved_pose_with_covariance = data
        self.observation = np.array([data.pose.pose.position.x, data.pose.pose.position.y])
        self.Swkm = np.array([[data.pose.covariance[0], 0], [0, data.pose.covariance[7]]])
        if not self.initialized:
            self.X = np.array([data.pose.pose.position.x, data.pose.pose.position.y, 0.0])
            self.initialized = True
        if (self.observation[0] == -10000.0) and (self.observation[1] == -10000.0):
            self.update(True)
        else:
            self.update(False)

    def prediction(self, xkm, U, Sigma_km1_km1):
        A, B = self.getGrad(xkm,U)
        x_new = np.matmul(A,xkm) + np.matmul(B,U)
        sigma_u = np.matmul(B ,np.matmul(self.Q, B.T))
        SxkGkm = np.matmul(A, np.matmul(Sigma_km1_km1, A.T)) + sigma_u
        return x_new, SxkGkm

    def correction(self, x_predict, Sx_k_km1, z_k, KalGain):
        x_new = x_predict + np.matmul(KalGain, (z_k - np.matmul(self.C, x_predict)))
        sigma_new = np.matmul((np.identity(3) - np.matmul(KalGain, self.C)), Sx_k_km1)
        return x_new, sigma_new

    def update(self, propagate_without_correct):

        X_predicted, Sx_k_km1 = self.prediction(self.X, self.U, self.Sx_k_k)                        
        X_corrected = X_predicted
        if not propagate_without_correct:
            self.gainUpdate(Sx_k_km1)
            X_corrected, self.Sx_k_k = self.correction(X_predicted, Sx_k_km1, self.observation, self.KalGain)   
        
        print("X k-1: {}".format(self.X))
        print("X_predicted: {}".format(X_predicted))
        print("X_corrected: {}".format(X_corrected))

        self.X = X_corrected
        self.X_corrected = np.reshape(X_corrected, [3, 1])

        self.receieved_pose_with_covariance.pose.pose.position.x = self.X[0]
        self.receieved_pose_with_covariance.pose.pose.position.y = self.X[1]
        self.measurement_pub.publish(self.receieved_pose_with_covariance)

    def gainUpdate(self, Sx_k_km1):
        self.KalGain = np.matmul(np.matmul(Sx_k_km1, self.C.T), (np.matmul(self.C, np.matmul(Sx_k_km1, self.C.T))) + self.Swkm)
        return

    def getGrad(self, x, U):
        R = 0.1
        L = 0.2

        # dxk = (r_wheel / 2 * (wri + wli) * math.cos(x[2]))
        # dyk = (r_wheel / 2 * (wri + wli) * math.sin(x[2]))
        # dthk = (r_wheel / l * (wri - wli))

        # Jacobian Matrices using above equations for motion model
        # A = [ df_x/dx  df_x/dy  df_x/dtheta ]
        #     [ df_y/dx  df_y/dy  df_y/dtheta ]
        #     [ df_theta/dx  df_theta/dy  df_theta/dtheta ]

        # B = [ df_x/du_1  df_x/du_2 ]
        #     [ df_y/du_1  df_y/du_2 ]
        #     [ df_theta/du_1  df_theta/du_2 ]

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
