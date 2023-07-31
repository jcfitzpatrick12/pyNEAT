'''
small class for holding different base movement functions
'''


'''
import packages
'''
import numpy as np
from sys_vars import sys_vars

#from one_pendulum_funcs import one_pendulum_funcs

class base_movement_functions:
     def __init__(self):
          self.sys_vars=sys_vars()
          self.base_movement_functions_dict = self.build_base_movement_functions_dict()

     def build_base_movement_functions_dict(self,):
          base_movement_functions_dict={}
          #O stands for oscillation
          base_movement_functions_dict['O']=[self.oscil_X,self.oscil_Xdotdot]
          #OD stands for oscillation and decay
          base_movement_functions_dict['OD'] = [self.oscil_decay_X,self.oscil_decay_Xdotdot]
          #CA stands for constant acceleration
          base_movement_functions_dict['CA'] = [self.const_acc_X,self.const_acc_Xdotdot]
          return base_movement_functions_dict

     def return_base_movement_function(self,requested_function):
          X = self.base_movement_functions_dict[requested_function][0]
          Xdotdot = self.base_movement_functions_dict[requested_function][1]
          return X,Xdotdot

     def oscil_X(self,t,**kwargs):
          return self.sys_vars.X_0*np.cos(self.sys_vars.big_omega*t)
     def oscil_Xdotdot(self,t,**kwargs):
          return -self.sys_vars.X_0*(self.sys_vars.big_omega**2)*np.cos(self.sys_vars.big_omega*t)

     def oscil_decay_X(self,t,**kwargs):
          return self.sys_vars.X_0*np.cos(self.sys_vars.big_omega*t)*np.exp(-1*self.sys_vars.k*t)
     def oscil_decay_Xdotdot(self,t,**kwargs):
          return self.sys_vars.X_0*(-np.exp(-1*self.sys_vars.k*t)*self.sys_vars.big_omega**2*np.cos(self.sys_vars.big_omega*t)+2*self.sys_vars.k*np.exp(-1*self.sys_vars.k*t)*self.sys_vars.big_omega*np.sin(self.sys_vars.big_omega*t)+self.sys_vars.k**2*np.exp(-1*self.sys_vars.k*t)*np.cos(self.sys_vars.big_omega*t))

     def const_acc_X(self,t,**kwargs):
          default_force_constant=0
          requested_force_constant=kwargs.get('force_constant',default_force_constant)
          return (requested_force_constant/2)*t**2
     def const_acc_Xdotdot(self,t,**kwargs):
          default_force_constant=0
          requested_force_constant=kwargs.get('force_constant',default_force_constant)
          return requested_force_constant

        



