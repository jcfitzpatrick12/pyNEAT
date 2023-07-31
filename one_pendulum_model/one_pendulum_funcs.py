import numpy as np
from sys_vars import sys_vars
from one_pendulum_model.base_movement_functions import base_movement_functions

class OnePendulumFuncs:
    #initalising the parameters
    def __init__(self,**kwargs):
        self.sys_vars=sys_vars()
        self.force_constant = kwargs.get('force_constant',0)

    def Xdotdot_at_t(self,t,requested_system,**kwargs):
        X,Xdotdot = base_movement_functions().return_base_movement_function(requested_system)
        Xdotdot_at_t=Xdotdot(t,**kwargs)
        return Xdotdot_at_t


    def general_system(self,y,t,requested_system,**kwargs):
        #unpacking the vector y
        theta, omega = y
        #return the C constant
        Xdotdot_at_t=self.Xdotdot_at_t(t,requested_system,**kwargs)
        #definining our system of ODE's
        dydt = [omega, -((self.sys_vars.g/self.sys_vars.l)*np.sin(theta) + (1/self.sys_vars.l)*np.cos(theta)*Xdotdot_at_t)]
        return dydt  

    #the pendulum system must be a function of two variables, y and t.
    #thus, use a lambda function so we can input which function we need!
    def return_pend_system(self,requested_system,**kwargs):
        return lambda y,t: self.general_system(y,t,requested_system,**kwargs)



