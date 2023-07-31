import numpy as np
from sys_vars import sys_vars
from one_genome_functions.feedforward import feedforward
from one_pendulum_model.balance_pendulum import BalancePendulum
from numpy.random import normal

class PendulumBalancingFitnessFuncs:
    def __init__(self):
        self.sys_vars=sys_vars()

    #for simplicity, we adjust the fitness of g, by simply dividing by the number of genomes that reside in the same species of g
    def find_adjusted_fitness(self,genome_collection):
        #sharing_sum = compatability_distance().sharing_sum
        #for each species list of genome_descriptors, loop through each genome_descriptor and update the "network output" key with the output vectors
        for genome_descriptors in genome_collection.genomes_dict.values():
                #if that species exists
                if len(genome_descriptors)!=0:
                    #then loop through each genome and compute the adjusted fitness
                    for genome_descriptor in genome_descriptors:
                         g = genome_descriptor['genome']
                         #sh_sum = sharing_sum(g,genome_collection)
                         #print(sh_sum,len(genome_descriptors))
                         #find the adjusted fitness
                         genome_descriptor['adjusted fitness']=genome_descriptor['fitness']/len(genome_descriptors)
        return genome_collection  
    

    #fill in the output vectors and compute the network fitnesses.
    def find_network_fitness(self,genome_collection):

        #y0 = [np.pi-0.1,0]
        y0 = [np.pi-normal(0,sys_vars().initial_theta_normal_width),normal(0,sys_vars().initial_theta_dot_normal_width)]
        #for each species list of genome_descriptors, loop through each genome_descriptor and update the "network output" key with the output vectors
        for genome_descriptors in genome_collection.genomes_dict.values():
                for genome_descriptor in genome_descriptors:
                    #extract the nth genome
                    g = genome_descriptor['genome']
                    #try balance the pendulum, find the time the pendulum fell
                    network_fitness,time_fell = BalancePendulum().try_balance(g,y0)  
                    #set the fitness to the time it took for the pendulum to fall
                    genome_descriptor['fitness']=network_fitness
                    genome_descriptor['time balanced']=time_fell

        #return (order preserving!) network outputs so that the ith element of XOR network outputs corresponds to the ith genome
        return genome_collection
