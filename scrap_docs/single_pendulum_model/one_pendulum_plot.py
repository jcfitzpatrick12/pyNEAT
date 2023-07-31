'''
pendulum plotting class. Given our numerically evaluated solution at the requested times, we can find

-second attempt to better incorporate a general X_t
'''

#class which handles plotting the numerically evaluated solutions.
#inputs are the solution and the sample times and plots the states of the pendulum

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from one_pendulum_funcs import one_pendulum_funcs
from base_movement_functions import base_movement_functions

#class which handles plotting the numerically evaluated solutions.
#inputs are the solution and the sample times and plots the states of the pendulum
#also inherits one_pendulum_funcs for the base function!
class one_pendulum_plot:
    def __init__(self,one_pendulum_funcs,sol,t,y0):
        self.one_pendulum_funcs = one_pendulum_funcs
        self.sol=sol
        self.t=t
        self.y0=y0
    
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
    
    #return big_theta*np.cos(self.big_omega*self.t)
    #verify is if we are checking the analytical solution
    def plot_sol(self,verify=None):
        #creating the figure
        fig,axs=plt.subplots(2,sharex=True)
        #adding x and y labels to each subplot
        #print(np.shape(self.sol))
        theta,omega = self.sol[:,0],self.sol[:,1]
        axs[0].set_ylabel(r'$\theta(t)$ [rad]')
        axs[0].plot(self.t,theta,color='black')
        axs[1].set_ylabel(r'$\omega(t)$ [rad $s^{-1}$]')
        axs[1].set_xlabel('$t$ [s]')
        axs[1].plot(self.t,omega,color='black')
        #if this function is just called normally to inspect the solution show plot now
        #otherwise (as if it is used in check_small_angle_sol) we will show the plot once we plot the analytical solution
        if verify==None:
            plt.show()
        return fig,axs
    
    def small_angle_analytical_sol(self):
        #for abbreviation purposes
        opf=self.one_pendulum_funcs
        theta_0=self.y0[0]
        small_omega = np.sqrt(opf.g/opf.l)
        big_theta = (opf.big_omega**2*opf.X_0)/(opf.l*(small_omega**2-opf.big_omega**2))
        return (theta_0-big_theta)*np.cos(small_omega*self.t)+big_theta*np.cos(opf.big_omega*self.t)
        #raise SystemExit
        #pass
    
    def check_small_angle_sol(self):
        fig,axs = self.plot_sol(True)
        small_angle_sol=self.small_angle_analytical_sol()
        axs[0].plot(self.t,small_angle_sol,color='grey',linestyle='--')
        plt.show()
        pass
    
        
    #plot all the snapshots of the pendulum
    def plot_pendulum_snapshots(self):
        #for abbreviation purposes
        opf=self.one_pendulum_funcs
        #initalizing the base movement function class, passes in the current instance of one_pendulum_funcs
        bmfs = base_movement_functions(opf)
        #extracting the requested base movement function (or more specifically for this application, it's second derivative)
        X_t,f_t = bmfs.movement_functions()
        fig,ax = self.create_pendulum_bare_plot()
        #finding the position of the base at each time 
        base_pos_x = X_t(self.t)
        #y position is always 0, obtain an array of the same shape but all zeros
        base_pos_y = base_pos_x*0
        #creating vectors that describe base_location
        base_locs = np.vstack((base_pos_x,base_pos_y))
        #unpack our numerically evaluated theta values from the solution
        theta = self.sol[:,0]
        #to plot the pendlum, recall our bob is always attached to the base at one point, and the other at
        #(as in notes) r(t)= (X(t) + l*np.sin(theta))e_x + l*cos(theta)e_y
        #x position of the bob
        r_t_x  = base_pos_x + opf.l*np.sin(theta)
        #y position of the bob
        r_t_y = -opf.l*np.cos(theta)
        #vector location of the bob at each time t
        #vstack means that bob_locs[0,:] holds the x locs of the bob and similatly for y locs
        bob_locs = np.vstack((r_t_x,r_t_y))

        #plotting bob and pendulum
        for i in range(0, len(self.t), 2):
            #print(0)
            #0 axis hold the x positions of the base and bob, 1 axis holds the y positions of the base and bob
            ax.plot([base_locs[0,i:i+2],bob_locs[0,i:i+2]], [base_locs[1,i:i+2],bob_locs[1,i:i+2]], 'bo-')
        
        plt.show()
        pass

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
            #for abbreviation purposes
            opf=self.one_pendulum_funcs
            #initalizing the base movement function class, passes in the current instance of one_pendulum_funcs
            bmfs = base_movement_functions(opf)
            #extracting the requested base movement function (or more specifically for this application, it's second derivative)
            X_t,f_t = bmfs.movement_functions()
            #creating the figure
            #fig,ax = self.create_pendulum_bare_plot()
            base_pos_x = X_t(frame)
            # y position is always 0, obtain an array of the same shape but all zeros
            base_pos_y = 0
            # extracting the index where the time instant lies in self.t
            n = np.where(self.t == frame)[0][0]
            # unpack our numerically evaluated theta values from the solution at this time instant
            theta = self.sol[n, 0]
            # x position of the bob at this instant
            bob_pos_x = base_pos_x + opf.l * np.sin(theta)
            bob_pos_y = -opf.l * np.cos(theta)
            # updating the base and bob position vectors
            ln.set_data([[base_pos_x, bob_pos_x]], [base_pos_y, bob_pos_y])
            #updating the time text
            annotation.set_text(str(np.round(frame,2)) + ' [s]')  # Update annotation text
            return ln, annotation,
    
        
        ani = FuncAnimation(fig, update, frames=self.t, init_func=init, blit=True)
        plt.show()

        pass