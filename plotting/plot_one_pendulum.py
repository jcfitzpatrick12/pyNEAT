'''
pendulum plotting class. Given our numerically evaluated solution at the requested times, we can find

-second attempt to better incorporate a general X_t
'''

#class which handles plotting the numerically evaluated solutions.
#inputs are the solution and the sample times and plots the states of the pendulum

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from one_pendulum_model.one_pendulum_funcs import OnePendulumFuncs
from one_pendulum_model.base_movement_functions import base_movement_functions
from sys_vars import sys_vars

#class which handles plotting the numerically evaluated solutions.
#inputs are the solution and the sample times and plots the states of the pendulum
#also inherits one_pendulum_funcs for the base function!
class one_pendulum_plot:
    def __init__(self,sol,t,base_pos_x,force_constants):
        self.sol=sol
        self.t=t
        self.base_pos_x = base_pos_x
        self.sys_vars=sys_vars()
        self.force_constants = force_constants

        #function to create the figure for the pendulum, deifning the bounds
    def create_pendulum_bare_plot(self):
        #creating the figure
        fig,ax=plt.subplots(1)
        #adding grid lines
        ax.grid()
        #set gridlines below
        ax.set_axisbelow(True)
        #setting axis limits
        ax.set_xlim(-1.2,1.2)
        ax.set_ylim(-1.2,1.2)
        #creating the bar that the base slides along
        ax.axhline(y=0,color='grey')
        #adding x and y labels
        ax.set_xlabel('$x$ [m]')
        ax.set_ylabel('$y$ [m]')
        return fig,ax

    def animate_pendulum(self):
        #create the bare plot
        fig,ax = self.create_pendulum_bare_plot()
        #initiate an (empty of data) plot function
        ln, = ax.plot([], [], 'bo-')
        #first time is naturally the zeroth entry of self.t
        time_str = str(self.t[0])+' [s]'
        #annotate the first time in a fixed location
        annotation = ax.annotate(time_str, xy=(0.65,0.05))
        #this will be updated each animation frame!
        annotation.set_animated(True)


        def init():
            return ln,annotation

        def update(frame):
            # y position is always 0, obtain an array of the same shape but all zeros
            base_pos_y = 0
            # extracting the index where the time instant lies in self.t
            n = np.where(self.t == frame)[0][0]
            # unpack our numerically evaluated theta values from the solution at this time instant
            theta = self.sol[n, 0]
            #and extract the base position at that time
            base_pos_x = self.base_pos_x[n]
            # x position of the bob at this instant
            bob_pos_x = base_pos_x + self.sys_vars.l * np.sin(theta)
            bob_pos_y = -self.sys_vars.l * np.cos(theta)
            # updating the base and bob position vectors
            ln.set_data([[base_pos_x, bob_pos_x]], [base_pos_y, bob_pos_y])
            #updating the time text
            annotation.set_text(str(np.round(frame,2)) + ' [s]')  # Update annotation text
            return ln, annotation,

        
        ani = FuncAnimation(fig, update, frames=self.t, init_func=init, blit=True,interval=5)
        plt.show()

        pass

    def plot_force_constant(self):
        plt.plot(self.t,self.force_constants[:len(self.t)])
        plt.xlabel('Time [s]')
        plt.ylabel(r'$\ddot X$ [$ms^{-1}$]')
        plt.show()
    
    #return big_theta*np.cos(self.big_omega*self.t)
    #verify is if we are checking the analytical solution
    def plot_sol(self,verify=None):
        #creating the figure
        fig,axs=plt.subplots(2,sharex=True)
        #adding x and y labels to each subplot
        #print(np.shape(self.sol))
        theta,omega = self.sol[:,0],self.sol[:,1]
        axs[0].set_ylabel(r'$\theta(t)$ [rad]')
        axs[0].axhline(y=np.pi/2,color='grey')
        axs[0].axhline(y=-np.pi/2,color='grey')
        axs[0].axhline(y=np.pi+self.sys_vars.balanced_tolerance,color='r')
        axs[0].axhline(y=np.pi,color='grey')
        axs[0].axhline(y=np.pi-self.sys_vars.balanced_tolerance,color='r')
        axs[0].plot(self.t,theta,color='black')
        axs[0].set_ylim(np.pi-self.sys_vars.balanced_tolerance-0.1,np.pi+self.sys_vars.balanced_tolerance+0.1)
        axs[1].set_ylabel(r'$\omega(t)$ [rad $s^{-1}$]')
        axs[1].set_xlabel('$t$ [s]')
        axs[1].plot(self.t,omega,color='black')
        #if this function is just called normally to inspect the solution show plot now
        #otherwise (as if it is used in check_small_angle_sol) we will show the plot once we plot the analytical solution
        if verify==None:
            plt.show()
        return fig,axs
    
    def small_angle_analytical_sol(self,y0):
        theta_0=y0[0]
        small_omega = np.sqrt(self.sys_vars.g/self.sys_vars.l)
        big_theta = (self.sys_vars.big_omega**2*self.sys_vars.X_0)/(self.sys_vars.l*(small_omega**2-self.sys_vars.big_omega**2))
        return (theta_0-big_theta)*np.cos(small_omega*self.t)+big_theta*np.cos(self.sys_vars.big_omega*self.t)
        #raise SystemExit
        #pass
    
    def check_small_angle_sol(self,y0):
        fig,axs = self.plot_sol(True)
        small_angle_sol=self.small_angle_analytical_sol(y0)
        axs[0].plot(self.t,small_angle_sol,color='grey',linestyle='--')
        plt.show()
        pass
    

     
