'''
Class which allows an input genome to try and balance a pendulum!
'''

from sys_vars import sys_vars
import numpy as np
from scipy.integrate import odeint
import numpy as np
from numpy.random import normal
from one_genome_functions.feedforward import feedforward

from one_pendulum_model.one_pendulum_funcs import OnePendulumFuncs
from one_pendulum_model.base_movement_functions import base_movement_functions

from general_functions.os_funcs import osFuncs

class BalancePendulum:
    def __init__(self):
        self.sys_vars=sys_vars()

    def find_base_pos_x(self,t,requested_movement_function,**kwargs):
        X,_ = base_movement_functions().return_base_movement_function(requested_movement_function,)
        base_pos_x = X(t,**kwargs)
        return base_pos_x
    
    def build_base_pos_x(self,x_distance_moved_list):
        #intiliase an array to hold the actual x position of the base at the different times
        base_pos_x_list = []
        #convert to a numpy array
        x_distance_moved_list=np.array(x_distance_moved_list)

        #I have to be careful as how I construct the base position x position list!
        for i in range(0,len(x_distance_moved_list)):
            if i ==0:
                base_pos_x_list.append(x_distance_moved_list[i])
            #list by list, translate each so that the "origin" is the final element of the previous list
            x_distance_moved_list[i]=x_distance_moved_list[i-1][-1]+x_distance_moved_list[i]
            #in this way, we gradually build up the bases actual position over time
            if i>0:
                base_pos_x_list.append(x_distance_moved_list[i][1:])
        #finally, concatenate the array
        base_pos_x = np.concatenate(base_pos_x_list)
        return base_pos_x
    
    def is_balanced(self,theta,k):
        #first assume the pendulum is balanced
        is_balanced=True
        #radians from the vertical
        rad_from_vertical = np.abs(np.pi-theta)
        #an array that tells us whether that angle is within our angle tolerance
        is_within_balanced_tolerance = rad_from_vertical<self.sys_vars.balanced_tolerance
        if not np.all(is_within_balanced_tolerance):
            is_balanced=False
            #find the timestep within the current interval the pendulum fell at!
            timestep_fell_within_interval = np.sum(is_within_balanced_tolerance)
            #index fell overall is the sum of
            # -the timestep it fell in the current interval
            # -number of timesteps in each interval it didn't fall, times k (the number of intervals we have evaluated without tipping)
            # -a correction that each k>0 we delete a duplicate entry
            index_fell = timestep_fell_within_interval + k*self.sys_vars.num_timesteps_for_sol - k
            return is_balanced,index_fell
        
        return is_balanced, -1

    def normalise_input_vector(self,input_vector):
        input_vector[1]/=self.sys_vars.balanced_tolerance
        input_vector[2]/=self.sys_vars.balanced_tolerance
        #print(input_vector)
        return input_vector

    def try_balance(self,g,y0):
        #call the solution at 101 evenly spaced intervals for 10 seconds
        t = np.linspace(0, self.sys_vars.response_time_interval, self.sys_vars.num_timesteps_for_sol)
        #also make a copy of the original interval
        t_original_interval = t.copy()
        
        #make an array to keep track of the different paremeters
        x_distance_moved_list = []
        force_constant_list = []
        theta_sols = []
        omega_sols = []
        tar=[]
        
        #define k to keep track of how many intervals we have evaluated!
        k=0
        while t[-1]<self.sys_vars.max_time:
            #build the input vector
            input_vector = [1,np.pi-y0[0],y0[1]]
            #normalise the input_vector
            input_vector=self.normalise_input_vector(input_vector)
            #find the network output
            network_output = feedforward().timestep_propagation(g,input_vector,10e-6,'tanh')[0]

            force_constant=self.sys_vars.force_multiple*network_output
            '''
            if force_constant<0:    
                print(force_constant)
            if force_constant>0:
                print(force_constant)           
            '''

            #print(force_constant)
            #append this to the rand force constant_list
            force_constant_list.append(np.ones(len(t))*force_constant)
            #find the system of equations that describe the pendulum dynamics
            system_tosolve = OnePendulumFuncs().return_pend_system('CA',force_constant=force_constant)
            #solve the system of equations
            sol = odeint(system_tosolve, y0, t)

            #find the distance moved over that interval (more flexible with in mind to extend to variable constant acceleration NN implementation)
            x_distance_moved = self.find_base_pos_x(t_original_interval,'CA',force_constant=force_constant)
            #append this to the base position
            x_distance_moved_list.append(x_distance_moved)

            #append the solution to the solution list (first element is superflous since we already have it
            #indexing 1: means we do not store duplicates
            #recall, our initial conditions were the final element of the previous integrations
            #unless of course, we are at our first integration!
            index_from = 0
            if k>0:
                index_from=1
            
            #thus, append our newly integrated solutions
            theta_sols.append(sol[index_from:,0])
            omega_sols.append(sol[index_from:,1])

            #build the next time interval over which to evaluate the solution
            t = np.linspace(t[-1],t[-1]+self.sys_vars.response_time_interval,self.sys_vars.num_timesteps_for_sol)
            tar.append(t[index_from:])

            '''
            check if the pendulum is still balanced!
            '''

            #takes in the interval we are on, and the solved theta values
            #this will allow us to determine if the pendulum is balanced or not, and when it fell, and the index of tar when it fell
            is_balanced, timestep_fell = self.is_balanced(sol[:,0],k)

            #if the pendulum is no longer balanced, break the while loop (no longer need to integrate)
            if not is_balanced:
                break
                
            '''
            update values for the next loop
            '''
            #update y0 to be the final solution values
            y0 = sol[-1]
            #increment k to keep track of which interval we are on
            k+=1



        #concatenate all the data
        conc_theta_sols = np.concatenate(theta_sols)
        conc_omega_sols = np.concatenate(omega_sols)
        conc_tar = np.concatenate(tar)
        conc_force_constants = np.concatenate(force_constant_list)

        #infer the base positions from the distance moved lists
        base_pos_x=self.build_base_pos_x(x_distance_moved_list) 

        #reassign the data to its usual shapes (some just reassign names to keep consistency with previous notation)
        sol = np.zeros((len(conc_tar),2))
        sol[:,0]=conc_theta_sols
        sol[:,1]=conc_omega_sols
        t=conc_tar


        '''
        find the time at which the pendulum fell!
        '''
        time_fell=t[timestep_fell]

        
        '''
        save the relevent data
        '''
        osFuncs().save_data('sol',sol)
        osFuncs().save_data('t',t)
        osFuncs().save_data('base_pos_x',base_pos_x)
        osFuncs().save_data('force_constants',conc_force_constants[:len(t)])

        


        network_fitness = self.network_fitness(time_fell,base_pos_x)
        return network_fitness,time_fell

    def network_fitness(self,time_fell,base_pos_x):
        #if the network goes outwith the track boundaries, return 0 fitness
        if np.any(np.abs(base_pos_x)>self.sys_vars.track_width):
            return 0
        network_fitness = time_fell
        #network_fitness = time_frac_penalty - radian_frac_penalty
        return network_fitness
