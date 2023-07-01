#import numpy
import numpy as np
#genome class handles functions relating to an individual genome g composed of connection genes and node genes
from mutation_functions import mutation_functions
#import genome builder to convert the hard coded edges into the adjacency matrices
from genome_builder import genome_builder

class test_mutation_example:
    def __init__(self):
        '''
        Hard code in some networks for verification
        '''
        #hard code in the node genes
        #our node_gene_indexing ordering convention is [input_nodes, output_nodes, hidden_nodes]
        #input nodes are labelled by -1, hidden nodes by 0 and output nodes by 1
        #so node_labels = [-1,...,-1,,1,...,1,0,...,0]
        #all nodes are assumed enabled
        self.test_node_genes = np.array([-1,-1,-1,1])
        #hard code in the connection genes of a simple pre-defined neural network
        #our connection gene convention is of the form [source_node_index,target_node_index,weight,enable_bit,innovation_number]
        self.test_connection_genes = np.array([[0,3,0.3,1],[1,3,1.3,1],[2,3,2.3,1]])

    def test_single_mutation(self):
        #creating a test genome 
        #test_genome = genome_builder().build_genome(self.test_node_genes,self.test_connection_genes,num_nodes=7)
        test_genome = genome_builder().build_genome(self.test_node_genes,self.test_connection_genes)
        '''
        now mutate the genome for one iteration
        '''
        #test the mutation function
        mutated_genome = mutation_functions().mutate(test_genome)
        return mutated_genome
