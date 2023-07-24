'''
class which handles evaluating the compatability difference as defined in the paper

"The number of excess and disjoint genes between a pair of genomes is a natural
measure of their compatibility distance. The more disjoint two genomes are, the less
evolutionary history they share, and thus the less compatible they are. Therefore, we
can measure the compatibility distance δ of different structures in NEAT as a simple lin-
ear combination of the number of excess E and disjoint D genes, as well as the average
weight differences of matching genes W , including disabled genes:
δ = c1E/N + c2D/N + c3 · W_bar "

where N is the number of genes in the larger genome, normalizes for genome size (N
can be set to 1 if both genomes are small, i.e., consist of fewer than 20 genes)
'''

import numpy as np
#directly draw out the uniform distribution sampling
from numpy.random import uniform,randint
from sys_vars import sys_vars
from two_genome_functions.two_genome_functions import TwoGenomeFunctions

class compatability_distance(TwoGenomeFunctions):
    def __init__(self,):
        self.sys_vars=sys_vars()

    def sharing_sum(self,genome,genome_collection):
        sharing_sum = 0
        for genome_descriptors in genome_collection.genomes_dict.values():
            for genome_descriptor in genome_descriptors:
                other_genome = genome_descriptor['genome']
                # Calculate the distance between g and other_genome
                distance = self.evaluate_distance(genome,other_genome)
                # Check if the distance is within the threshold
                if distance <= self.sys_vars.delta_t:
                    sharing_sum += 1
        return sharing_sum

    '''
    evaluate distance takes in two genomes and outputs their compatability distance
    '''
    def evaluate_distance(self,genome0,genome1):
        #create the combined connection-genes array.
        combined_connection_genes = TwoGenomeFunctions().combine_genome_connection_genes(genome0,genome1)
        #find the disjoint and excess genes
        #these occur if innovation number in one parent != the innovation number of another at the same edge element in the adjacency matrix
        #first we find the genes which contain either a disjoint or excess gene IN EITHER PARENT.
        #we will not yet verify which are disjoint and which are excess
        disjoint_or_excess_genes = TwoGenomeFunctions().find_disjoint_or_excess_genes(combined_connection_genes)
        #extract the number of dijoint or excess nodes in each parent
        num_disjoint_genome0,num_excess_genome0 = self.num_disjoint_and_excess(0,disjoint_or_excess_genes,combined_connection_genes)
        num_disjoint_genome1,num_excess_genome1 = self.num_disjoint_and_excess(1,disjoint_or_excess_genes,combined_connection_genes)
        #so we can find E and D simply
        E = num_excess_genome0+num_excess_genome1
        D = num_disjoint_genome0+num_disjoint_genome1
        #the average weight difference is also found simply

        matching_genes = TwoGenomeFunctions().find_matching_genes(combined_connection_genes)
        #NOTE we need the average weight difference of MATCHING genes, we have not done this!
        W_bar = np.mean(np.abs(combined_connection_genes[matching_genes,0,0]-combined_connection_genes[matching_genes,0,1])) 
        '''
        adapt in future to set N=1 if size of network is small
        '''
        #set N to be the maximum number of enabled nodes 
        #N=max(genome0.num_nodes(type='enabled'),genome1.num_nodes(type='enabled'))
        N=1
        #and thus we can find delta to be
        delta = (E*self.sys_vars.c1 + D*self.sys_vars.c2)/N + W_bar*self.sys_vars.c3
        return delta

    