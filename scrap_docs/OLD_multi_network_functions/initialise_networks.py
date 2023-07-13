'''
Class which contains a single function that will initialise N networks of a (user inputted) arbitrary topology and
initiales them with random weights.
Currently, the output is a python list of N genome objects, but we aim to convert this into one multi network genome class object
which includes an extra dimension to index the genome.
'''

from sys_vars import sys_vars
from one_genome_functions.genome_builder import genome_builder
from multi_network_functions.genome_collection import GenomeCollection
from numpy.random import randint,uniform
import numpy as np

class InitialiseNetwork:
    def __init__(self):
        self.sys_vars = sys_vars()

    #function which builds a python list of N genome objects, each of identical topology but random weights
    #the input node_genes labels each node
    #requested_edges takes in pairs of edges which define the network.
    #all edges are assumed enabled
    #all edges will be assigned a random weight from weight range.
    def build_genome_collection(self,num_networks,requested_node_labels,requested_edges):
        #first find out how many edges the user has requested.
        num_edges = len(requested_edges)
        #extract the weight range from sys_vars
        weight_range=self.sys_vars.weight_range
        #build the connection genes for the nth network
        connection_genes = np.zeros((num_edges,4))
        #place in the requested edges into connection_genes
        connection_genes[:,:2]=requested_edges
        #enable all the edges (by default)
        connection_genes[:,3]=np.ones(num_edges)
        #create an array that will hold the genome class object in each entry, for each network
        genomes = []
        #loop through each network and build the connection genes with randomly assigned weights at each iteration
        for n in range(num_networks):
            #create a set of random weights to build the connection genes for the nth network
            random_weights = uniform(weight_range[0],weight_range[1],num_edges)
            #reassign the random weights into connection_genes
            connection_genes[:,2] = random_weights
            #build the nth genome
            genome = genome_builder().build_genome(requested_node_labels,connection_genes)
            #append this genome to genome collection
            genomes.append(genome)


        #all genomes start in the same species! (indexed by zero)
        species_labels = np.zeros(num_networks)
        #return the class object which holds the genomes
        return GenomeCollection(genomes,species_labels)



