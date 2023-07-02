#import numpy
import numpy as np
#genome class handles functions relating to an individual genome g composed of connection genes and node genes
from mutation_functions import mutation_functions
#import genome builder to convert the hard coded edges into the adjacency matrices
from genome_builder import genome_builder
#import the rough visualisation code
from visualise_genome import visualise_genome
#import the mating functions
from mating import mating

#testing mating().mate()
class test_mate_example:
    def __init__(self):
        '''
        Hard code in the parent genes in the NEAT paper (figure 4) for verification
        '''
        #hard code in the node genes
        #our node_gene_indexing ordering convention is [input_nodes, output_nodes, hidden_nodes]
        #input nodes are labelled by -1, hidden nodes by 0 and output nodes by 1
        #so node_labels = [-1,...,-1,,1,...,1,0,...,0]
        #all nodes are assumed enabled
        self.parent0_node_genes = np.array([-1,-1,-1,-1,1,0])
        #hard code in the connection genes of a simple pre-defined neural network
        #our connection gene convention is of the form [source_node_index,target_node_index,weight,enable_bit,innovation_number]
        self.parent0_connection_genes = np.array([[1,4,1.4,1],[2,4,2.4,0],[3,4,3.4,1],[2,5,2.5,1],[5,4,5.4,1],[1,5,1.5,1]])
        #define parent2's node and connection_genes
        self.parent1_node_genes = np.array([-1,-1,-1,-1,1,0,0])
        self.parent1_connection_genes = np.array([[1,4,1.4,1],[2,4,2.4,0],[3,4,3.4,1],[2,5,2.5,1],[5,4,5.4,0],[5,6,5.6,1],[6,4,6.4,1],[3,5,3.5,1],[1,6,1.6,1]])



    def test_mate(self):

        #creating the parent genomes
        parent0 = genome_builder().build_genome(self.parent0_node_genes,self.parent0_connection_genes)
        parent1 = genome_builder().build_genome(self.parent1_node_genes,self.parent1_connection_genes)
        #visualise_genome().plot_network(parent1)
        #visualise_genome().plot_network(parent2)

        #artificially changing the connection_gene innovation numbers to match the paper
        parent0.connection_innov_numbers[1,5]=7
        parent1.connection_innov_numbers[3,5]=8
        parent1.connection_innov_numbers[1,6]=9
        #output the child
        child = mating().mate(parent0,parent1,1,2)
        return child