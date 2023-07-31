'''
small class for holding different base movement functions
'''


'''
import packages
'''
import numpy as np

#from one_pendulum_funcs import one_pendulum_funcs

class base_movement_functions:
    def __init__(one_pendulum_funcs):
        super().__init__()

    #function that returns the base movement functio X_t and it's second derivative f_t
    def movement_functions(self):

        '''
        #constant force on the base
        if opf.X_t_string=='scalar':
             def X_t(t,acceleration_constant):
                  return (acceleration_constant/2)*t**2
             def f(t,):
                  return C     
        '''

             

        #defining the oscillating base functions
        if self.sys_vars().X_t_string=='oscil':
                def X_t(t):
                    return self.sys_vars().X_0*np.cos(self.sys_vars().big_omega*t)
                def f_t(t):
                    return -self.sys_vars().X_0*(self.sys_vars().big_omega**2)*np.cos(self.sys_vars().big_omega*t)
        
        if self.sys_vars().X_t_string=='oscil_decay':
                def X_t(t):
                     return self.sys_vars().X_0*np.cos(self.sys_vars().big_omega*t)*np.exp(-1*self.sys_vars().k*t)
                
                def f_t(t):
                     return self.sys_vars().X_0*(-np.exp(-1*self.sys_vars().k*t)*self.sys_vars().big_omega**2*np.cos(self.sys_vars().big_omega*t)+2*self.sys_vars().k*np.exp(-1*self.sys_vars().k*t)*self.sys_vars().big_omega*np.sin(self.sys_vars().big_omega*t)+self.sys_vars().k**2*np.exp(-1*self.sys_vars().k*t)*np.cos(self.sys_vars().big_omega*t))
                
        #returning them
        return X_t,f_t
        



