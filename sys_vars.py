'''
class which neatly stores the mutation variables
'''

import numpy as np

class sys_vars:
    def __init__(self):

        '''
        mutation variables
        '''
        #the  range from which weights can be sampled
        self.requested_weight_distribution = "normal"
        #the weight range if self.requested_weight_distribution is set to uniform
        self.uniform_weight_range = [0,1]
        #the loc and width of a gaussian if requested_weight distirbution is set to normal
        self.normal_loc_width = [0,1.0]
        #the probability that a genomes weights will be perturbed
        self.probability_weight_perturbed = 0.75
        #each weight has this probability of it's weights being uniformly perturbed
        self.probability_uniform_weight_perturbation = 0.9
        #where the perturbation is sampled from a uniform distribution with radius defined below. So new_weight = old_weight + rv sampled form uniform distribution centred at zero of below radius
        self.uniform_perturbation_radius = 0.05
        #below variable not explicitly used but nice for clarity. Otherwise, the connection is assigned a new weight sampled from self.weight_range
        self.probability_new_weight_value = 1-self.probability_uniform_weight_perturbation
        #probability that with a given mutation of a network, that a new node will be added
        self.probability_add_node = 0.01
        #self.probability_add_node = 1.0
        #a "small network" has a total number of nodes bounded by the below (i.e if num_total_nodes<=small_network size => network is a small network)
        self.small_network_size = 20
        #probability a new link will be added to a small network at a particular mutation
        self.probability_add_link_small_network = 0.05
        #probability a new link will be added to a large network (if network is not small => network is large)
        self.probability_add_link_large_network = 0.3
        #whether we are going to allow cycles when adding a link
        self.allow_cycles=False

        '''
        mating variables
        '''
        #probability that we enable a gene if it was disabled in either parent
        self.probability_enable_gene = 0.25

        '''
        compatability evaluation variables
        '''
        #constants which modify the sensitivity of the compatability distance on the number of disjoint, excess and the mean weight difference
        self.c1=1.0
        self.c2 = 1.0
        self.c3 = 0.15
        #constant which defines the threshold to compatability to which genomes belong to a species
        self.delta_t = 3.0

        '''
        reproduction variables
        '''
        self.parents_lower_bound_prob = 0.25
        #remove this percent of the lowest performing networks at reproduction
        self.num_lowest_performs_removed_pc = 0.1
        #boolean to tell us whether we will copy the best performers in each species to the next generation
        self.keep_champions = True
        #if so, how many will we keep? as a percentage
        self.num_of_champions_pc = 0.1

        '''
        NEAT variables
        '''
        #max number of generation iteration
        self.max_generation=50
        #in how many generations, if the max fitness of a species has not increased, then it is not allowed to reproduce
        self.stagnant_index= 15

        '''
        single pendulum balancing variables
        '''
        #the half length of the track (base cannot go further than this)
        self.track_width = 0.5
        #maximum time the simulation will run for
        self.max_time = 180
        #the balancing time
        self.balance_time = 120
        #the tolerance angle which we define the pendulum to be balanced for! around 15 deg in radians
        self.balanced_tolerance = 0.5
        #length of time the network controls the base for (seconds) before evaluating a new acceleration for the base
        self.response_time_interval = 0.25
        #number of timesteps we need a solution at (evenly spaced)
        self.num_timesteps_for_sol = 20

        #gaussian widths of the distributions we sample the initial conditions from
        self.initial_theta_normal_width = 0.025
        self.initial_theta_dot_normal_width = 0.0

        #multiple of the network output to get the actual acceleration of the base
        self.force_multiple = 0.75


        '''
        pendulum testing parameters
        '''

        #grav constant [ms^-2]
        self.g=9.8
        #l is the length of the pendulum [m]
        self.l=0.75
        #X_0 is the oscillation amplitude of the base [m]
        self.X_0 = 0.025
        #big omega is the oscillation frequency of the base [Hz]
        self.big_omega = 20
        #decay constant
        self.k=0.05
        #constant acceleration constant
        self.force_constant=1.0
        #how long simulation will run [s]
        self.T = 5
        #take n_t equidistant samples of the solution in time
        self.n_t = self.T*50
        #consider an initial position
        self.y0 = [0.0, 0.0] #this is a list NOT an array
        #which base movement function are we considering? O - oscillating, OD - oscillating and decay, CA - constant acceleration
        self.analytical_test = 'O'