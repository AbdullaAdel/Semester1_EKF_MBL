from GaussianFilter import *
import numpy as np
class EKF(GaussianFilter):
    """
    Extended Kalman Filter class. Implements the :class:`GaussianFilter` interface for the particular case of the Extended Kalman Filter.
    """
    def __init__(self, x0, P0, *args):
        """
        Constructor of the EKF class.

        :param x0: initial mean state vector
        :param P0: initial covariance matrix
        :param args: arguments to be passed to the parent class
        """
        super().__init__(x0, P0, *args)  # call parent constructor

    def f(self, xk_1, uk): # motion model
        """
        Motion model of the EKF **to be overwritten by the child class**.

        :param xk_1: previous mean state vector
        :param uk: input vector
        :return xk_bar, Pk_bar: predicted mean state vector and its covariance matrix
        """
        pass

    def Jfx(self, xk_1):
        """
        Jacobian of the motion model with respect to the state vector. **Method to be overwritten by the child class**.

        :param xk_1: Linearization point. By default the linearization point is the previous state vector taken from a class attribute.
        :return: Jacobian matrix
        """
        pass

    def Jfw(self, xk_1):
        """
        Jacobian of the motion model with respect to the noise vector. **Method to be overwritten by the child class**.

        :param xk_1: Linearization point. By default the linearization point is the previous state vector taken from a class attribute.
        :return: Jacobian matrix
        """
        pass

    def h(self, xk):  # observation model
        """
        The observation model of the EKF is given by:

        .. math::
            z_k=h(x_k,v_k)
            :label: eq-EKF-observation-model

        This method computes the mean of this direct observation model. Therefore it does not depend on v_k since it is
        a zero mean Gaussian noise.

        :param xk: mean of the predicted state vector. By default it is taken from the class attribute.
        :return: expected observation vector
        """
        pass

    def Prediction(self, uk, Qk, xk_1=None, Pk_1=None):
        """
        Prediction step of the EKF. It calls the motion model and its Jacobians to predict the state vector and its covariance matrix.

        :param uk: input vector
        :param Qk: covariance matrix of the noise vector
        :param xk_1: previous mean state vector. By default it is taken from the class attribute. Otherwise it updates the class attribute.
        :param Pk_1: covariance matrix of the previous state vector. By default it is taken from the class attribute. Otherwise it updates the class attribute.
        :return xk_bar, Pk_bar: predicted mean state vector and its covariance matrix. Also updated in the class attributes.
        """
        # logging for plotting
        self.xk_1 = xk_1 if xk_1 is not None else self.xk_1
        self.Pk_1 = Pk_1 if Pk_1 is not None else self.Pk_1

        self.uk = uk
        self.Qk = Qk  # store the input and noise covariance for logging
        
        # KF equations begin here
        # TODO: To be implemented by the student

        xk_bar = self.f(xk_1, uk)
        A_k = self.Jfx(xk_1, uk)
        W_k = self.Jfw(xk_1)
        
        Pk_bar = A_k @ Pk_1 @ A_k.T + W_k @ Qk @ W_k.T
        

        return xk_bar, Pk_bar
    

    def Update(self, zk, Rk, xk_bar, Pk_bar, Hk, Vk):
        """
        Update step of the EKF. It calls the observation model and its Jacobians to update the state vector and its covariance matrix.

        :param zk: observation vector
        :param Rk: covariance matrix of the noise vector
        :param xk_bar: predicted mean state vector.
        :param Pk_bar: covariance matrix of the predicted state vector.
        :param Hk: Jacobian of the observation model with respect to the state vector.
        :param Vk: Jacobian of the observation model with respect to the noise vector.
        :return xk,Pk: updated mean state vector and its covariance matrix. Also updated in the class attributes.
        """
        # logging for plotting
        self.xk_bar = xk_bar
        self.Pk_bar = Pk_bar
        self.zk = zk
        self.nz = zk.shape[0];  # store dimensionality of the observation
        self.Rk = Rk  # store the observation and noise covariance for logging


        # KF equations begin here

        # TODO: To be implemented by the student
        # if self.k < 2000:
        #     return xk_bar, Pk_bar
        if zk.shape[0] == 0 :
            print("Before Here")
            return xk_bar, Pk_bar
        
        if zk.shape[0] == 2:
            print("here")
        # Compute the innovation covariance
        S = Hk @ Pk_bar @ Hk.T + Vk @ Rk @ Vk.T
        # Compute the Kalman gain without reshaping:
        K_k = Pk_bar @ Hk.T @ np.linalg.pinv(S)

        # Update the state:
        # xk = xk_bar + K_k @ (zk - Hk @ xk_bar)
        xk = xk_bar + K_k @ (zk - self.h(xk_bar))
        print(zk - self.h(xk_bar))

        # Update the covariance using the Joseph form:
        I = np.eye(Pk_bar.shape[0])
        Pk = (I - K_k @ Hk) @ Pk_bar @ (I - K_k @ Hk).T 
        # K_k = Pk_bar @ Hk.T @ np.linalg.pinv(Hk @ Pk_bar @ Hk.T + Vk @ Rk @ Vk.T)
        print(f'Iteration number{self.k}')
        

        # xk = xk_bar + K_k @ (zk - Hk @ xk_bar)
        # # xk = xk.reshape(3, 1)
        # Pk = (np.eye(Hk.shape[1]) - K_k @ Hk) @ Pk_bar @ (np.eye(Hk.shape[1]) - K_k @ Hk).T
        # Pk = Pk.reshape(3, 3)
        self.Pk = Pk
        self.xk = xk
        return xk, Pk
