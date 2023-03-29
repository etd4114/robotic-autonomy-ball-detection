import numpy as np
# import human_walk
import matplotlib.pyplot as plt
import pdb
import rospy
from geometry_msgs.msg import PoseWithCovariance, Point, Pose, PoseWithCovarianceStamped
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
import math
import numpy as np


class EKF():
    def __init__(self, nk, dt, X, U):
        self.nk = nk
        self.dt = 1
        self.X = X
        self.U = U

        # <--------<< Initialize corection covariance
        self.Sigma_init = np.array([[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]])
        # <--------<< Should be updated with variance from the measurement
        self.sigma_measure = np.array(
            [[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]])
        # <--------<< Initialize Kalman Gain
        self.KalGain = np.random.rand(3, 3)

        # <--------<< Subscribe to the ball pose topic
        self.measurement_sub = rospy.Subscriber(
            "/pose_covariance", PoseWithCovariance, self.measurement_cb)
        self.measurement_pub = rospy.Publisher(
            "/pose_corrected", PoseWithCovariance, queue_size=10)
        # print(self.z_k)

        self.dl = 0.1
        self.Sx_k_k = self.Sigma_init
        self.mxkmGkm = np.array([0, 0, 0])
        self.Swkm = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        pose_with_cov = PoseWithCovariance(
            pose=Pose(
                position=Point(
                    x=0,
                    y=0,
                    z=0
                )
            ))
        self.receieved_pose_with_covariance = pose_with_cov

    def measurement_cb(self, data):
        self.receieved_pose_with_covariance = data
        self.X = [data.pose.position.x,
                  data.pose.position.y, data.pose.position.z]

    def prediction(self, x, U, Sigma_km1_km1):
        # TODO
        # Use the motion model and input sequence to predict a n step look ahead trajectory.
        # You will use only the first state of the sequence for the rest of the filtering process.
        # So at a time k, you should have a list, X = [xk, xk_1, xk_2, xk_3, ..., xk_n] and you will use only xk.
        # The next time this function is called a lnew list is formed.
        dmxkmGkm = self.dotX(x, U)
        mxkGkm = x + dt*dmxkmGkm
        A = self.getGrad(self.mxkmGkm, self.dl, U)
        SxkGkm = np.dot(np.dot(A, Sigma_km1_km1), A) + self.Swkm
        return mxkGkm, SxkGkm

    def correction(self, x_predict, Sx_k_km1, z_k, KalGain):

        I = np.diag(np.ones((1, len(x_predict))))
        C = self.getGrad(x_predict, self.dl, KalGain)
        mxkGk = x_predict + np.dot(self.KalGain, (z_k - x_predict))
        SxkGk = np.dot((I - self.KalGain*C),Sx_k_km1)
        return mxkGk, SxkGk
        # TODO
        # Write a function to correct your prediction using the observed state.

    def update(self):
        self.X_pred = self.X
        X_predicted, Sx_k_km1 = self.prediction(
            self.X, self.U, self.Sx_k_k)                        # PREDICTION STEP
        X_corrected, self.Sx_k_k = self.correction(
            X_predicted, Sx_k_km1, self.X, self.KalGain)   # CORRECTION STEP
        # GAIN UPDATE
        self.gainUpdate(Sx_k_km1)
        self.X = X_corrected

        self.X_pred = np.reshape(self.X_pred, [3, 1])
        self.X_corrected = np.reshape(
            X_corrected, [3, 1])   # <--------<< Publish

        self.receieved_pose_with_covariance.pose.position.x = self.X[0]
        self.receieved_pose_with_covariance.pose.position.y = self.X[1]
        self.receieved_pose_with_covariance.pose.position.z = self.X[2]
        self.measurement_pub.publish(self.receieved_pose_with_covariance)

    def gainUpdate(self, Sx_k_km1):
        C = np.array(self.getGrad(Sx_k_km1, self.dl, self.U))
        f_1 = np.dot(Sx_k_km1, C.T)
        f_2 = 1 / (C*Sx_k_km1*C.T + self.U + 10e8)
        self.KalGain = np.dot(f_1, f_2)
        # TODO
        # Write a function to update the Kalman Gain, a.k.a. self.KalGain

    def dotX(self, x, U):
        dxk = []
        dxk.append(U[0])
        dxk.append(U[1])
        dxk.append(U[2])
        dxk = np.array(dxk)
        return dxk * x
        # TODO
        # This is where your motion model should go. The differential equation.
        # This function must be called in the self.predict function to predict the future states.

    def getGrad(self, x, dl, U):
        xs = len(x)
        for i in range(xs):
            dx = np.zeros((xs, 1))
            dx[i] = dl/2
            x1 = x-dx
            x2 = x+dx
            f1 = self.dotX(x1, U)
            f2 = self.dotX(x2, U)
        return (f2 - f1)/dl
        # TODO
        # Linearize the motion model here. It should be called in the self.predict function and should yield the A and B matrix.


if __name__ == '__main__':
    rospy.init_node("EKF")

    # ---------------Define initial conditions --------------- #
    nk = 1       # <------<< Look ahead duration in seconds
    dt = 0.1       # <------<< Sampling duration of discrete model
    X = np.array([0, 0, 0])       # <------<< Initial State of the Ball
    U = np.array([0, 0, 0])       # <------<< Initial input to the motion model

    filter = EKF(nk, dt, X, U)
    for i in range(5):
        filter.update()
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        filter.update()
        rate.sleep()
