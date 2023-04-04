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
        self.sigma_measure = np.array(
            [[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]])
        # <--------<< Initialize Kalman Gain
        self.KalGain = np.random.rand(3, 3)

        # <--------<< Subscribe to the ball pose topic
        self.measurement_sub = rospy.Subscriber(
            self.measurement_sub_topic, PoseWithCovariance, self.measurement_cb)
        self.measurement_pub = rospy.Publisher(
            self.measurement_pub_topic, PoseWithCovariance, queue_size=10)
        self.pub_visualize_ball = rospy.Publisher(
            self.published_ball_visualize_topic, Marker, queue_size=10)
        self.dl = 0.1
        self.Sx_k_k = self.Sigma_init
        self.mxkmGkm = np.array([0, 0, 0])
        self.Swkm = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.observation = np.array([0,0,0])
        self.thetaXY = 0
        self.receieved_pose_with_covariance = PoseWithCovariance()
        self.seq = 0
        self.frame_id = "/camera_frame"
        self.init_Marker()
        
    def measurement_cb(self, data):
        self.receieved_pose_with_covariance = data
        euler = quaternion2euler([data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w])
        self.current_posesition_roll_angle = euler[1]
        self.observation = np.array([data.pose.position.x,
                  data.pose.position.y, self.current_posesition_roll_angle])
        self.Swkm = [data.covariance[0],
                     data.covariance[7],
                     data.covariance[14]]
        self.update()

    def prediction(self, xkm, U, Sigma_km1_km1):
        # TODO
        # Use the motion model and input sequence to predict a n step look ahead trajectory.
        # You will use only the first state of the sequence for the rest of the filtering process.
        # So at a time k, you should have a list, X = [xk, xk_1, xk_2, xk_3, ..., xk_n] and you will use only xk.
        # The next time this function is called a lnew list is formed.
        x, y, theta = self.dotX(xkm, U)
        xk = [0,0,0]
        xk[0] = xkm[0] + self.dt*x
        xk[1] = xkm[1] + self.dt*y
        xk[2] = xkm[2] + self.dt*theta
        A = self.getGrad(xkm, self.dl, U)
        SxkGkm = np.dot(np.dot(A, Sigma_km1_km1), A) + self.Swkm
        return xk, SxkGkm

    def correction(self, x_predict, Sx_k_km1, z_k, KalGain):

        I = np.diag(np.ones((1, len(x_predict))))
        C = self.getGrad(x_predict, self.dl, self.U)
        mxkGk = x_predict + np.dot(self.KalGain, (z_k - x_predict))
        SxkGk = np.dot((I - self.KalGain*C),Sx_k_km1)
        return mxkGk, SxkGk
        # TODO
        # Write a function to correct your prediction using the observed state.

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
        X_predicted, Sx_k_km1 = self.prediction(
            self.X, self.U, self.Sx_k_k)                        # PREDICTION STEP
        X_corrected, self.Sx_k_k = self.correction(
            X_predicted, Sx_k_km1, self.observation, self.KalGain)   # CORRECTION STEP
        # GAIN UPDATE
        self.gainUpdate(Sx_k_km1)
        # self.X.append(X_corrected)
        print("X k-1: {}".format(self.X))
        print("X_predicted: {}".format(X_predicted))
        print("X_corrected: {}".format(X_corrected))
        self.X = X_corrected
        self.X_pred = np.reshape(self.X_pred, [3, 1])
        self.X_corrected = np.reshape(
            X_corrected, [3, 1])   # <--------<< Publish

        quat = euler2quaternion([0,self.X[2],0])
        self.receieved_pose_with_covariance.pose.position.x = self.X[0]
        self.receieved_pose_with_covariance.pose.position.y = self.X[1]
        self.receieved_pose_with_covariance.pose.orientation.x = quat[0]
        self.receieved_pose_with_covariance.pose.orientation.y = quat[1]
        self.receieved_pose_with_covariance.pose.orientation.z = quat[2]
        self.receieved_pose_with_covariance.pose.orientation.w = quat[3]
        self.measurement_pub.publish(self.receieved_pose_with_covariance)
        self.ball.pose = self.receieved_pose_with_covariance.pose
        self.pub_visualize_ball.publish(self.ball)


    def gainUpdate(self, Sx_k_km1):
        C = np.array(self.getGrad(Sx_k_km1, self.dl, self.U))
        f_1 = np.dot(Sx_k_km1, C.T)
        f_2 = 1 / (C*Sx_k_km1*C.T + self.U + 10e-8)
        self.KalGain = np.dot(f_1, f_2)
        # TODO
        # Write a function to update the Kalman Gain, a.k.a. self.KalGain

    def dotX(self, x, U):
        # differential motion model
        l = 0.2
        r_wheel = 0.1
        wli = U[0]
        wri = U[1]
        thk = U[2]
        
        dxk = r_wheel / 2 * (wri + wli) * math.cos(thk)
        dyk = r_wheel / 2 * (wri + wli) * math.sin(thk)
        dthk = r_wheel / l * (wri - wli)
        return np.array([dxk, dyk, dthk])
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
    U = np.array([1, 0.9, 0.1])       # <------<< Initial input to the motion model

    filter = EKF(nk, dt, X, U)
    for i in range(5):
        filter.update()
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        # filter.update()
        rate.sleep()
