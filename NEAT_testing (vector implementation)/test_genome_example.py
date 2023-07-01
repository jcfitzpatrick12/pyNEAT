#import numpy
import numpy as np
#genome class handles functions relating to an individual genome g composed of connection genes and node genes
#from genome_functions import genome_functions
#import the class that defines the genome of each network
from genome_builder import genome_builder
#import genome functions which holds the functions which act on a single genome
#from genome_functions import genome_functions
#import the rough visualisation code
from visualise_genome import visualise_genome


class test_genome_example:
    def __init__(self):
        '''
        Hard code in some networks for verification
        '''
        #hard code in the node genes
        #our node_gene_indexing ordering convention is [input_nodes, output_nodes, hidden_nodes]
        #input nodes are labelled by -1, hidden nodes by 0 and output nodes by 1
        #so node_labels = [-1,...,-1,,1,...,1,0,...,0]
        #all nodes are assumed enabled
        #self.test_node_genes = np.array([[0,-1],[1,1],[2,1],[3,0],[4,0],[5,0]])
        self.test_node_genes = np.array([-1,1,1,0,0,0])
        #hard code in the connection genes of a simple pre-defined neural network
        #our connection gene convention is of the form [source_node_index,target_node_index,weight,enable_bit,innovation_number]
        #but the innovation number is implicately assigned based on the order of the connection_genes
        self.test_connection_genes = np.array([[0,3,0.3,1],[0,4,0.4,1],[0,5,0.5,1],[3,2,3.2,1],[4,1,4.1,1],[5,1,5.1,1]])

    def test(self):
        k=0
        #create the initially padded test_genome
        test_genome = genome_builder().build_genome(self.test_node_genes,self.test_connection_genes,num_nodes=8)
        #check out some genes indexed by k
        print(test_genome.connection_genes[...,k])
        print(test_genome.node_genes)
        #return the genome with only enabled indices
        test_genome = test_genome.return_enabled_genome()
        #check out some genes indexed by k
        print(test_genome.connection_genes[...,k])
        print(test_genome.node_genes)
        #return the original padded genome
        test_genome=test_genome.return_padded_genome(num_nodes=8)
        #check out some genes indexed by k
        print(test_genome.connection_genes[...,k])
        print(test_genome.node_genes)
        pass