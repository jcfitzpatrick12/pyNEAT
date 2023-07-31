'''
class which defines the base movement functions, and the system of equations describing the pendulum state
for odeint to solve
'''


'''
import packages
'''
import numpy as np
import matplotlib.pyplot as plt
from base_movement_functions import base_movement_functions
'''
defining the one_pendulum_functions which we use to numerically evaluate the solution
'''
class one_pendulum_funcs:
    #initalising the parameters
    def __init__(self,sys_vars):
        self.sys_vars=sys_vars()
       

    #let the vector y=(theta,omega) describe the state of the pendulum
    #let f(t) [f_t] be the second derivative of the movement function X(t) [X_t]


    #defining the system of equations we need to solve
    def pend_system(self,y,t,):
        #initalizing the base movement function class, passes in the current instance of this class as input
        bmfs = base_movement_functions(self)
        #extracting the requested base movement function (or more specifically for this application, it's second derivative)
        X_t,f_t = bmfs.movement_functions()
        #unpacking the vector y
        theta, omega = y
        #second derivative of the movement function X(t), which we denote f_t
        C = f_t(t)
        #definining our system of ODE's
        dydt = [omega, -((self.sys_vars.g/self.sys_vars.l)*np.sin(theta) + (1/self.sys_vars.l)*np.cos(theta)*C)]
        return dydt
    




