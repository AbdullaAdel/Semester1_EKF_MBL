from GFLocalization import *
from EKF import *
from DR_3DOFDifferentialDrive import *
from DifferentialDriveSimulatedRobot import *

class EKF_3DOFDifferentialDriveCtVelocity(GFLocalization, DR_3DOFDifferentialDrive, EKF):

    def __init__(self, kSteps, robot, *args):

        self.x0 = np.zeros((6, 1))  # initial state x0=[x y z psi u v w r]^T
        self.P0 = np.zeros((6, 6))  # initial covariance

        # this is required for plotting
        self.index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1),
                 IndexStruct("u", 3, 2), IndexStruct("v", 4, 3), IndexStruct("yaw_dot", 5, None)]
        
        # TODO: To be completed by the student
        super().__init__(index, kSteps, robot, x0, P0, *args)
        
        self.t_1 = 0
        self.t = 0
        self.Dt = self.t - self.t_1

    def f(self, xk_1, uk):
        # TODO: To be completed by the student
        etak_1 = xk_1[0:3]
        nuk_1 = xk_1[3:6] # Hardcoded for now
        
        etak_bar = Pose3D.oplus(etak_1, nuk_1 * self.dt)
        nuk_bar = nuk_1 
        xk_bar = np.array([etak_bar, nuk_bar]).reshape(6,1)
         
        return xk_bar # size 6x1 [x , y , theta , u , v , w]

    def Jfx(self, xk_1, uk=None):
        # TODO: To be completed by the student
        etak_1 = np.array(xk_1[0:3])
        nuk_1 = np.array(xk_1[3:6])
        
        a = Pose3D.J_1oplus(etak_1, nuk_1*self.dt)
        b =  Pose3D.J_2oplus(etak_1) * self.dt
        c = np.zeros((3,3))
        d = np.eye(3)
        J = np.block([[a, b], [c, d]]) # Should be 6x6
        return J

    def Jfw(self, xk_1):
        # TODO: To be completed by the student
        a = Pose3D.J_2oplus(xk_1[0:3])*(self.dt**2)/2
        b = np.eye(3)*self.dt
        J = np.vstack((a,b))
        return J

    def h(self, xk):  #:hm(self, xk):
        # TODO: To be completed by the student
        h = self.Jhx(xk)
        return h * xk # return the expected observations
        # zk_dist contains [displacement, angle]
        
    def Jhx(self, zk):
        zk = np.where(zk != 0, 1, 0)
        print(zk)
        h = np.array([[0,0,zk[0][0],0,0,0],
                      [0,0,0,zk[1][0],0,0],
                      [0,0,0,0,0,0],
                      [0,0,0,0,0,zk[3][0]]])
                      
        return h
    def Jhw(self, xk):
        h = np.eye(4) # number of measurements = 3
        return h
    def GetInput(self):
        """

        :return: uk,Qk:
        """
        # TODO: To be completed by the student
        # Return None for the displacement and return error for the velocity
        
        uk = None
        Qk = self.robot.Qsk

        # uk = np.array([[0],[0],[0]])
        
        return uk, Qk

    def GetMeasurements(self):  # override the observation model
        """

        :return: zk, Rk, Hk, Vk
        """
        # TODO: To be completed by the student

        zk_dist, R_dist = DR_3DOFDifferentialDrive.GetInput(self) # two measurements (leveraging the code that was written for the input as a measurement)
        # measurement are [displacement of robot, angle of robot]
        # change it to displacement in x and y ?


        # zk_dist contains [displacement, angle]
        displacement = zk_dist[0][0]  # Extract displacement
        angle = zk_dist[1][0]         # Extract angle

        # Transform to Cartesian coordinates
        delta_x = displacement 
        delta_y = 0 
        
        # z_k_cartesian = np.array([[displacement , 0 , angle]]).T
        
        z_k_cartesian = np.array([[displacement , 0 , angle]]).T
        z_k_cartesian /= self.dt
        
        # Compute the Jacobian of the transformation
        J_movement = np.array([[1,0],
                               [0,0],
                               [0,1]])
        

# Transform the covariance

        R_dist_cartesian = J_movement @ R_dist @ J_movement.T
        R_dist_cartesian /= self.dt
        
        zk_yaw, R_yaw = self.robot.ReadCompass() # one measurements

        
        if zk_dist[0][0] is None:
            zk_dist[0][0] = 0
        if zk_dist[1][0] is None:
            zk_dist[1][0] = 0
        if zk_yaw is None:
            zk_yaw = 0
        # stacking the measurements
        zk_yaw = np.array([[zk_yaw]]).reshape(1,1)
        R_yaw = np.array([[R_yaw]]).reshape(1,1)
        
        
        zk = np.vstack((zk_yaw,z_k_cartesian))
        # print(f"R_dist = {R_dist.shape} R_yaw = {R_yaw.shape} zk = {zk.shape}")
        Rk = np.block([
        [R_yaw, np.zeros((1, 3))],
        [np.zeros((3, 1)), R_dist_cartesian]
]       )
        
        # returns observation choice
        Hk = self.Jhx(zk) # TODO test point
        Vk = self.Jhw(1)
        Vk = Vk
        return zk, Rk, Hk, Vk


if __name__ == '__main__':

    M = [CartesianFeature(np.array([[-40, 5]]).T),
           CartesianFeature(np.array([[-5, 40]]).T),
           CartesianFeature(np.array([[-5, 25]]).T),
           CartesianFeature(np.array([[-3, 50]]).T),
           CartesianFeature(np.array([[-20, 3]]).T),
           CartesianFeature(np.array([[40,-40]]).T)]  # feature map. Position of 2 point features in the world frame.

    xs0 = np.zeros((6,1))  # initial simulated robot pose

    robot = DifferentialDriveSimulatedRobot(xs0, M)  # instantiate the simulated robot object
    kSteps = 4000

    xs0 = np.zeros((6, 1))  # initial simulated robot pose
    index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1),
                 IndexStruct("u", 3, 2), IndexStruct("v", 4, 3), IndexStruct("yaw_dot", 5, None)]

    x0 = np.array([[0.0, 0.0, 0.0, 0.5, 0.0, 0.03]]).T
    P0 = np.diag(np.array([0.0, 0.0, 0.0, 0.5 ** 2, 0 ** 2, 0.05 ** 2]))

    dd_robot = EKF_3DOFDifferentialDriveCtVelocity(kSteps, robot)  # initialize robot and KF
    dd_robot.LocalizationLoop(x0, P0, np.array([[0.5, 0.03]]).T)  # run localization loop

    exit(0)