'''
class which neatly stores the mutation variables
'''

from numpy.random import uniform,normal

class sys_vars:
    def __init__(self):

        '''
        mutation variables
        '''
        #the  range from which weights can be sampled
        self.requested_weight_distribution = "normal"
        self.uniform_weight_range = [0,1]
        self.normal_loc_width = [0,0.75]
        #the probability that a genomes weights will be perturbed
        self.probability_weight_perturbed = 0.85
        #each weight has this probability of it's weights being uniformly perturbed
        self.probability_uniform_weight_perturbation = 0.9
        #where the perturbation is sampled from a uniform distribution with radius defined below. So new_weight = old_weight + rv sampled form uniform distribution centred at zero of below radius
        self.uniform_perturbation_radius = 0.05
        #below variable not explicitly used but nice for clarity. Otherwise, the connection is assigned a new weight sampled from self.weight_range
        self.probability_new_weight_value = 1-self.probability_uniform_weight_perturbation
        #probability that with a given mutation of a network, that a new node will be added
        self.probability_add_node = 0.0025
        #self.probability_add_node = 1.0
        #a "small network" has a total number of nodes bounded by the below (i.e if num_total_nodes<=small_network size => network is a small network)
        self.small_network_size = 20
        #probability a new link will be added to a small network at a particular mutation
        self.probability_add_link_small_network = 0.075
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
        #remove this percent of the lowest performing networks at reproduction
        self.num_lowest_performs_removed_pc = 0.1
        #boolean to tell us whether we will copy the best performers in each species to the next generation
        self.keep_champions = True
        #if so, how many will we keep?
        self.num_of_champions_pc = 0.1
        #self.num_of_champions = 5

        '''
        NEAT variables
        '''
        #max number of generation iteration
        self.max_generation=50
        #in how many generations, if the max fitness of a species has not increased, then it is not allowed to reproduce
        self.stagnant_index= 15