#import numpy
import numpy as np
#genome class handles functions relating to an individual genome g composed of connection genes and node genes
from genome_functions import genome_functions
#import genome builder to convert the hard coded edges into the adjacency matrices
from genome import genome_builder
#import the class that defines the genome of each network
from genome import genome
#import the rough visualisation code
from visualise_genome import visualise_genome

class test_mutation_example:
    def __init__(self):
        '''
        Hard code in an initial network to mutate. 
        '''
        #hard code in the node genes
        #our node_gene_indexing convention is [bias, input_nodes, output_nodes, hidden_nodes]
        #this structure will mean it is e
        #input nodes are labelled by -1, hidden nodes by 0 and output nodes by 1
        #for the mutation test, the inital network must have ONLY input and output nodes
        #start simple with three input nodes and one output node
        self.test_node_genes = np.array([[0,-1],[1,-1],[2,-1],[3,1]])
        #hard code in the connection genes of a simple pre-defined neural network
        #our connection gene convention is of the form [source_node_index,target_node_index,weight,enable_bit,innovation_number]
        self.test_connection_genes = np.array([[0,3,0.3,1,0],[1,3,1.3,1,1],[2,3,2.3,1,2]])

    def test_single_mutation(self):
        #creating a test genome 
        test_genome = genome_builder().build_genome(self.test_node_genes,self.test_connection_genes,num_nodes=6)
        #visualise the genome before mutation
        #visualiser = visualise_genome(test_genome)
        #visualiser.plot_network()
        #genome_functions handles the operations on the network, given the defined genome
        test_genome_functions = genome_functions(test_genome)
        '''
        now mutate the genome for one iteration
        '''
        #test the mutation function
        mutated_genome = test_genome_functions.mutate()
        #visualise the genome after a single mutation
        visualiser = visualise_genome(mutated_genome)
        visualiser.plot_network()
        pass
