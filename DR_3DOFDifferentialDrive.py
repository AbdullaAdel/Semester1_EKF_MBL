from Localization import *
import numpy as np
from DifferentialDriveSimulatedRobot import *
from Feature import *
from Pose import *

class DR_3DOFDifferentialDrive(Localization):
    """
    Dead Reckoning Localization for a Differential Drive Mobile Robot.
    """
    def __init__(self, index, kSteps, robot, x0, *args):
        """
        Constructor of the :class:`prlab.DR_3DOFDifferentialDrive` class.

        :param args: Rest of arguments to be passed to the parent constructor
        """

        super().__init__(index, kSteps, robot, x0, *args)  # call parent constructor

        self.dt = 0.1  # dt is the sampling time at which we iterate the DR
        self.t_1 = 0.0  # t_1 is the previous time at which we iterated the DR
        self.wheelRadius = 0.1  # wheel radius
        self.wheelBase = 0.5  # wheel base
        self.robot.pulse_x_wheelTurns = 4096  # number of pulses per wheel turn

    def Localize(self, xk_1, uk):  # motion model
        """
        Motion model for the 3DOF (:math:`x_k=[x_{k}~y_{k}~\psi_{k}]^T`) Differential Drive Mobile robot using as input the readings of the wheel encoders (:math:`u_k=[n_L~n_R]^T`).

        :parameter xk_1: previous robot pose estimate (:math:`x_{k-1}=[x_{k-1}~y_{k-1}~\psi_{k-1}]^T`)
        :parameter uk: input vector (:math:`u_k=[u_{k}~v_{k}~w_{k}~r_{k}]^T`)
        :return xk: current robot pose estimate (:math:`x_k=[x_{k}~y_{k}~\psi_{k}]^T`)
        """

# TODO Verify if inputs match the expected ones
        uk = uk[0]
        print(f'uk = \n\n{uk}')
        print(f'xk_1 in Localize = \n\n {xk_1} ')
        xk = np.zeros([3,1])  # initialize the pose vector
        
        

        # Calculate total velocity and angular velocity
        v = uk[0][0]
        w = uk[1][0]

        # Update the pose vector
        xk[2][0] = xk_1[2][0] + w
        
        xk[0][0] = xk_1[0][0] + v*np.cos(xk[2][0])
        xk[1][0] = xk_1[1][0] + v*np.sin(xk[2][0])
        
        return xk



    def GetInput(self):
        """
        Get the input for the motion model. In this case, the input is the readings from both wheel encoders.

        :return: uk:  input vector (:math:`u_k=[n_L~n_R]^T`)
        """
        # Reading the encoders as a measurement that is a gaussian. zsk = [n_L~n_R]^T, Re = diag(σ_L^2,σ_R^2)
        # zsk is a mean vector and Re is a covariance matrix
        zsk , Re = self.robot.ReadEncoders()

        # Calculating the  mean distance per pulse, and transforming the covariance matrix from pulse to distance
        x_L = (zsk[0]/self.robot.pulse_x_wheelTurns)*2*np.pi*self.wheelRadius
        x_R = (zsk[1]/self.robot.pulse_x_wheelTurns)*2*np.pi*self.wheelRadius
        
        J_p2x = np.array([[(2*np.pi*self.wheelRadius)/self.robot.pulse_x_wheelTurns,0],
                          [0,(2*np.pi*self.wheelRadius)/self.robot.pulse_x_wheelTurns]])
        
        Rx = J_p2x @ Re @ J_p2x.T
        
        # Calculating the mean displacement and angular displacement and transforming the covariance matrix
        disp = 0.5 * (x_L + x_R)
        a_disp = (x_L - x_R) / self.wheelBase
        
        
        A = np.array([[0.5],
                      [0.5]])
        B = np.array([[1/self.wheelBase],
                      [-1/self.wheelBase]])
        RxL = np.array([[Rx[0][0]]])
        RxR = np.array([[Rx[1][1]]])
        Qk = A @ RxL @ A.T + B @ RxR @ B.T
        
        # Noise sampled from the Gaussian distribution N(0,Qk) After transformation from pulse to distance
        # traveled per wheel to x,dtheta in the B frame
        noise_x , noise_dtheta = np.random.multivariate_normal([0,0], Qk)
        disp += noise_x
        a_disp += noise_dtheta

        
        uk = np.array([[disp],[a_disp]])


        
        return uk, Qk

if __name__ == "__main__":

    # feature map. Position of 2 point features in the world frame.
    M = [CartesianFeature(np.array([[-40, 5]]).T),
           CartesianFeature(np.array([[-5, 40]]).T),
           CartesianFeature(np.array([[-5, 25]]).T),
           CartesianFeature(np.array([[-3, 50]]).T),
           CartesianFeature(np.array([[-20, 3]]).T),
           CartesianFeature(np.array([[40,-40]]).T)]  # feature map. Position of 2 point features in the world frame.
    np.random.seed(0)
    xs0=np.zeros((6,1))   # initial simulated robot pose
    robot = DifferentialDriveSimulatedRobot(xs0, M) # instantiate the simulated robot object

    kSteps = 5000 # number of simulation steps
    xsk_1 = xs0 = np.zeros((6, 1))  # initial simulated robot pose
    index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1)] # index of the state vector used for plotting

    x0=Pose3D(np.zeros((3,1)))
    dr_robot=DR_3DOFDifferentialDrive(index,kSteps,robot,x0)
    dr_robot.LocalizationLoop(x0, np.array([[0.5, 0.03]]).T)

    exit(0)