#import numpy
import numpy as np
#genome class handles functions relating to an individual genome g composed of connection genes and node genes
#from genome_functions import genome_functions
#import the class that defines the genome of each network
from genome import genome_builder
#import genome functions which holds the functions which act on a single genome
from genome_functions import genome_functions


class test_feedforward_example:
    def __init__(self):
        '''
        Hard code in some networks for verification
        '''
        #hard code in the node genes
        #our node_gene_indexing convention is [bias, input_nodes, output_nodes, hidden_nodes]
        #this structure will mean it is e
        #input nodes are labelled by -1, hidden nodes by 0 and output nodes by 1
        self.test_node_genes = np.array([[0,-1],[1,1],[2,1],[3,0],[4,0],[5,0]])
        #hard code in the connection genes of a simple pre-defined neural network
        #our connection gene convention is of the form [source_node_index,target_node_index,weight,enable_bit,innovation_number]
        self.test_connection_genes = np.array([[0,3,0.3,1,0],[0,4,0.4,1,1],[0,5,0.5,1,2],[3,2,3.2,1,3],[4,1,4.1,1,4],[5,1,5.1,1,5]])
        #input vector to the NN (one for each input node)
        self.test_input_vector = np.array([1])
        #the hand computed outputs of node 1 and 2
        self.correct_output = np.array([4.19,0.96])

    def test(self):
        #creating a test genome 
        test_genome = genome_builder().build_genome(self.test_node_genes,self.test_connection_genes,num_nodes = 8)
        #create the class which handles functions on this genome
        test_genome_functions = genome_functions(test_genome)
        #find the output of our neural network given our input vector
        output_vector=test_genome_functions.feedforward(self.test_input_vector,10e-6,'linear')

        #testing the feedforward_function
        #find the output of our neural network given our input vector

        print(f'The correct output is: {self.correct_output}. The computed output is {output_vector}')
        #if the computed output is (to a very small tolerance) identical to the correct output
        if np.abs(np.sum(self.correct_output-output_vector)) < 10e-10:
            print('Feedforward verified.')
        else:
            raise ValueError(f'Feedforward not verified: Expected {self.correct_output}, got {output_vector}')       

        pass
