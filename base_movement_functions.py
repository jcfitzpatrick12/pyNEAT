'''
small class for holding different base movement functions
'''


'''
import packages
'''
import numpy as np
import matplotlib.pyplot as plt

#from one_pendulum_funcs import one_pendulum_funcs

class base_movement_functions:
    def __init__(self,one_pendulum_funcs):
        #taking in one_pendulum_funcs class as input 
        self.one_pendulum_funcs = one_pendulum_funcs

    #function that returns the base movement functio X_t and it's second derivative f_t
    def movement_functions(self):
        #abbreviate for clarity
        opf = self.one_pendulum_funcs
        #raise SystemExit
        #defining the oscillating base functions
        if opf.X_t_string=='oscil':
                def X_t(t):
                    return opf.X_0*np.cos(opf.big_omega*t)
                def f_t(t):
                    return -opf.X_0*(opf.big_omega**2)*np.cos(opf.big_omega*t)
        
        if opf.X_t_string=='oscil_decay':
                def X_t(t):
                     return opf.X_0*np.cos(opf.big_omega*t)*np.exp(-1*opf.k*t)
                
                def f_t(t):
                     return opf.X_0*(-np.exp(-1*opf.k*t)*opf.big_omega**2*np.cos(opf.big_omega*t)+2*opf.k*np.exp(-1*opf.k*t)*opf.big_omega*np.sin(opf.big_omega*t)+opf.k**2*np.exp(-1*opf.k*t)*np.cos(opf.big_omega*t))
                
        #returning them
        return X_t,f_t
        



