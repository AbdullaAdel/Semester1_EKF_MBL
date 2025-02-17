from GFLocalization import *
from EKF import *
from DR_3DOFDifferentialDrive import *
from DifferentialDriveSimulatedRobot import *

class EKF_3DOFDifferentialDriveInputDisplacement(GFLocalization, DR_3DOFDifferentialDrive, EKF):
    """
    This class implements an EKF localization filter for a 3 DOF Diffenteial Drive using an input displacement motion model incorporating
    yaw measurements from the compass sensor.
    It inherits from :class:`GFLocalization.GFLocalization` to implement a localization filter, from the :class:`DR_3DOFDifferentialDrive.DR_3DOFDifferentialDrive` class and, finally, it inherits from
    :class:`EKF.EKF` to use the EKF Gaussian filter implementation for the localization.
    """
    def __init__(self, kSteps, robot, *args):
        """
        Constructor. Creates the list of  :class:`IndexStruct.IndexStruct` instances which is required for the automated plotting of the results.
        Then it defines the inital stawe vecto mean and covariance matrix and initializes the ancestor classes.

        :param kSteps: number of iterations of the localization loop
        :param robot: simulated robot object
        :param args: arguments to be passed to the base class constructor
        """

        self.dt = 0.1  # dt is the sampling time at which we iterate the KF
        x0 = np.zeros((3, 1))  # initial state x0=[x y z psi u v w r]^T
        P0 = np.zeros((3, 3))  # initial covariance

        # this is required for plotting
        index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("z", 2, 0), IndexStruct("yaw", 3, 1)]

        self.t_1 = 0
        self.t = 0
        self.Dt = self.t - self.t_1
        super().__init__(index, kSteps, robot, x0, P0, *args)

    def f(self, xk_1, uk):
        # TODO: To be completed by the student
        
        xk_bar = Pose3D.oplus(xk_1, uk)
        return xk_bar

    def Jfx(self, xk_1, uk):
        # TODO: To be completed by the student
        J = Pose3D.J_1oplus(xk_1, uk)
        return J

    def Jfw(self, xk_1):
        # TODO: To be completed by the student
        J = Pose3D.J_2oplus(xk_1)
        return J

    def h(self, xk):  #:hm(self, xk):
        # TODO: To be completed by the student
        h = xk[2]
        return h  # return the expected observations

    def GetInput(self):
        """

        :return: uk,Qk
        """
        # TODO: To be completed by the student
        # uk = [dx, dphi], Qk = Covariance. 
        # The input is in terms of velocity and angular velocity, and its covariance 
        Uk , Re = DR_3DOFDifferentialDrive.GetInput(self)

        # Changing the input from dx,dphi to dx,dy,dphi which is to ensure that the size of the matrix final is (3,1)
        
        dx = Uk[0,0]
        dy = 0
        dphi = Uk[1,0] 
        
        # the B frame only moves along the x axis and there is no movement in the y axis
        J_new = np.array([[1,0],
                          [0,0],
                          [0,1]])
        Qk = J_new @ Re @ J_new.T
        
        uk = np.array([[dx],[dy],[dphi]])
        uk = uk.reshape(3,1)
        
        return uk, Qk

    def GetMeasurements(self):  # override the observation model
        """
        Reads compass measurements and generates the observation vector (zk),
        its covariance (Rk), and the Jacobians (Hk, Vk).

        :return: zk, Rk, Hk, Vk
        """

        # Read compass measurement
        z_yaw, sigma2_yaw = self.robot.ReadCompass()

        # Check if a valid measurement is available
        if z_yaw is not None:
            # Add compass measurement to observation vector
            zk = np.array([[z_yaw]])
            zk = zk.reshape(1, 1)
            Rk = np.diag(sigma2_yaw)
            Hk = np.array([[0, 0, 1]])  # Observation Jacobian for yaw
            Hk = Hk.reshape(1, 3)
            Vk = np.eye(1)  # Noise Jacobian for yaw

        # Ensure correct default return when no measurements are available
        if z_yaw is None:
            return None, None, None, None

        return zk, Rk, Hk, Vk



if __name__ == '__main__':

    M = [CartesianFeature(np.array([[-40, 5]]).T),
           CartesianFeature(np.array([[-5, 40]]).T),
           CartesianFeature(np.array([[-5, 25]]).T),
           CartesianFeature(np.array([[-3, 50]]).T),
           CartesianFeature(np.array([[-20, 3]]).T),
           CartesianFeature(np.array([[40,-40]]).T)]  # feature map. Position of 2 point features in the world frame.

    np.random.seed(2)
    xs0 = np.zeros((6,1))  # initial simulated robot pose

    robot = DifferentialDriveSimulatedRobot(xs0, M)  # instantiate the simulated robot object
    kSteps = 5000

    xs0 = np.zeros((6, 1))  # initial simulated robot pose
    index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1)]

    x0 = np.zeros((3, 1))
    P0 = np.zeros((3, 3))

    dd_robot = EKF_3DOFDifferentialDriveInputDisplacement(kSteps,robot)  # initialize robot and KF
    dd_robot.LocalizationLoop(x0, P0, np.array([[0.5, 0.03]]).T)

    exit(0)