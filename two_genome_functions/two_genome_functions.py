'''
function that holds the functions that act between two arbitrary genomes.
These functions are used in e.g. mating.py and compatability_distance.py
'''

import numpy as np
#directly draw out the uniform distribution sampling
from numpy.random import uniform,randint
from sys_vars import sys_vars
from one_genome_functions.genome import genome

class TwoGenomeFunctions:
    def __init__(self):
        self.sys_vars=sys_vars()

    def return_padded_pair(self,genome0,genome1):
            #pad out the smaller network so we can better match up the two directly encoded networks
            #first find the parent with the largest number of nodes.
            num_global_nodes_genome0 = genome0.num_nodes(type='global_max')
            num_global_nodes_genome1 = genome1.num_nodes(type='global_max')
            #if genome 0 is larger , pad genome 1
            if num_global_nodes_genome0>num_global_nodes_genome1:
                genome1=genome1.return_padded_genome(num_nodes=num_global_nodes_genome0)
            #if genome 1 is larger , pad genome 0
            elif num_global_nodes_genome1>num_global_nodes_genome0:
                genome0=genome0.return_padded_genome(num_nodes=num_global_nodes_genome1)
            return genome0,genome1

    #function which returns the combined genomes connection genes (padding where necessary so both genomes genes are the same shape, even if the networks have different topologies)
    #of dimension (num_global_max_nodes_both_genomes,num_global_max_nodes_both_genomes,3,2)
    #where the zeroth element last axis is genome0 and the first element last axis is genome1
    def combine_genome_connection_genes(self,genome0,genome1):
        #pad the smaller genome
        genome0,genome1=self.return_padded_pair(genome0,genome1)
        #now both genomes have arrays of the same shapes!
        #make a combined matrix of the connection_genes (so we can efficiently extract which gene to take from which using indices)
        #place genome0 into the 0 element of the last axis
        #place genome1 into the 1 element of the last axis
        combined_connection_genes = np.zeros(np.shape(genome0.connection_genes)+(2,))
        combined_connection_genes[...,0]=genome0.connection_genes
        combined_connection_genes[...,1]=genome1.connection_genes
        return combined_connection_genes

    def find_disjoint_or_excess_genes(self,combined_connection_genes):
        #take the pairwise difference in innovation numbers between both genomes
        innov_difference_between_genomes = np.diff(combined_connection_genes[...,2,:],axis=-1)[...,0]
        #if this is non-zero we have a disjoint or excess gene in at least one of the genomes (yet to be determined which one)
        disjoint_or_excess_in_either_genome = innov_difference_between_genomes != 0 
        #copy this over the last axis for pairwise multiplication
        #print(np.shape(disjoint_or_excess_in_either_genome))
        disjoint_or_excess_in_either_genome_vect = np.repeat(disjoint_or_excess_in_either_genome[...,np.newaxis],2,axis=-1)
        #find the proposed excess and disjoint genes in both genomes (true if it exists at the edge element)
        #this splits them back into the respective genomes!
        disjoint_or_excess_genes = np.multiply(combined_connection_genes[...,2,:],disjoint_or_excess_in_either_genome_vect)>0
        #different genomes are indexed over the last axis. Now the only non-zero_innovation numbers are those which are disjoint
        return disjoint_or_excess_genes
    
    def find_matching_genes(self,combined_connection_genes):
        #take the pairwise difference in innovation numbers between both genomes
        innov_difference_between_genomes = np.diff(combined_connection_genes[...,2,:],axis=-1)[...,0]
        #if this is non-zero we have a disjoint or excess gene in at least one of the genomes (yet to be determined which one)
        matching_genes = innov_difference_between_genomes == 0 
        return matching_genes


    #takes in the genomes index, where the disjoint or excess genes are for that genome, and the combined connection_genes
    def num_disjoint_and_excess(self,requested_genome_index,disjoint_or_excess_genes,combined_connection_genes):
        #find the other genomes index
        other_genome_index = 1-requested_genome_index
        max_innov_other_genome= np.nanmax(combined_connection_genes[...,2,other_genome_index])
        #print(max_innov_other_genome)
        #extract the connection genes of the requested genomes index
        requested_genome_connection_genes = combined_connection_genes[...,requested_genome_index]
        #extract specifically disjoint or excess genes (only the innovation numbers)
        #print(np.shape(disjoint_or_excess_genes))
        #print(np.shape(requested_genome_connection_genes))
        disjoint_or_excess_gene_innovation_numbers = requested_genome_connection_genes[disjoint_or_excess_genes[:,:,requested_genome_index],2]
        #an array which holds ones for only disjoint genes
        disjoint_genes = disjoint_or_excess_gene_innovation_numbers<=max_innov_other_genome
        excess_genes = disjoint_or_excess_gene_innovation_numbers>max_innov_other_genome
        #return the number of disjoint and excess genes respectively
        return np.nansum(disjoint_genes),np.nansum(excess_genes)