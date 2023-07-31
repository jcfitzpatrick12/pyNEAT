'''
Class which contains a single function that will initialise N networks of a (user inputted) arbitrary topology and
initiales them with random weights.
Currently, the output is a python list of N genome objects, but we aim to convert this into one multi network genome class object
which includes an extra dimension to index the genome.

How do we keep track of the ordering of network outputs and network fitnesses?
I will rewrite the code so that genome_collection.genome_dict is so that each dictionary element is:

species_index: [genome1_descriptor,genome2_descriptor,...,genomeN_descriptor]

where N is the number of genomes in the species labelled by species_index
and genome1_descriptor is itself a dictionary where each dictionary elements are 

"genome": genome class object
"fitness": genome fitness evaluated using output vector
"adjusted_fitness": the adjusted fitness computed following speciation.

is this convoluted?

I don't think so, we could then see that over the course of one generation, we build each item sequentially,

genome -> output_vector -> genome_fitness (sort into new species using genome compatability, but move the entire descriptor!
this way, we can keep track of each networks fitness and there is no discrepancy in the flattening ordering.) -> genome_adjusted_fitness

and then from genome_adjusted_fitness we build the new genomes and we start again.
'''

from one_genome_functions.genome_builder import genome_builder
from one_genome_functions.weight_distributions import WeightDistributions
from neat_functions.genome_collection import GenomeCollection
from plotting.plot_genome import visualise_genome
from numpy.random import uniform,normal

from sys_vars import sys_vars
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
        #build the connection genes for the nth network
        connection_genes = np.zeros((num_edges,4))
        #place in the requested edges into connection_genes
        connection_genes[:,:2]=requested_edges
        #enable all the edges (by default)
        connection_genes[:,3]=np.ones(num_edges)
        #create an array that will hold the genome class object in each entry, for each network
        genomes_dict = {}
        #create a list to hold all the genome descriptors (note all genomes start in the same species!)
        genome_descriptors = []

        #loop through each network and build the connection genes with randomly assigned weights at each iteration
        for n in range(num_networks):
            #create a set of random weights to build the connection genes for the nth network
            random_weights = WeightDistributions().return_weights(num_edges)
            #random_weights = normal(self.sys_vars.normal_weight_params[0],self.sys_vars.normal_weight_params[1],num_edges)
            #reassign the random weights into connection_genes
            connection_genes[:,2] = random_weights
            #build the nth genome
            genome = genome_builder().build_genome(requested_node_labels,connection_genes)
            #visualise_genome().plot_network(genome)
            #raise SystemExit
            #build the genome_descriptor dictionary
            genome_descriptor = {}
            #build the descriptor dictionary
            #start by placing in the genome
            genome_descriptor['genome']=genome
            #all other descriptors are so far None, as they have not been yet computed
            #genome_descriptor['network output']=None
            genome_descriptor['fitness']=None
            genome_descriptor['adjusted fitness']=None
            genome_descriptor['time balanced']=None
            genome_descriptor['champion']=False
            #append this genome descriptor to genome collection
            genome_descriptors.append(genome_descriptor)
        #print(genome_descriptors)
        #artificially impose three species, where the 1-labelled species is extinct
        genomes_dict[0]=genome_descriptors
        
        #return the class object which holds the genomes
        return GenomeCollection(genomes_dict)