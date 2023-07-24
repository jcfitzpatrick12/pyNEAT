#import numpy
import numpy as np
#genome class handles functions relating to an individual genome g composed of connection genes and node genes
#from genome_functions import genome_functions
#import the class that defines the genome of each network
from one_genome_functions.genome_builder import genome_builder
#import genome functions which holds the functions which act on a single genome
from one_genome_functions.mutation_functions import mutation_functions
#import the feedforward function
from one_genome_functions.feedforward import feedforward
from one_genome_functions.visualise_genome import visualise_genome


class test_feedforward_example:
    def __init__(self):
        '''
        Hard code in some networks for verification
        '''
        #hard code in the node genes
        #our node_gene_indexing ordering convention is [input_nodes, output_nodes, hidden_nodes]
        #input nodes are labelled by -1, hidden nodes by 0 and output nodes by 1
        #so node_labels = [-1,...,-1,,1,...,1,0,...,0]
        #all nodes are assumed enabled
        self.test_node_labels = np.array([-1,1,1,0,0,0])
        #hard code in the connection genes of a simple pre-defined neural network
        #our connection gene convention is of the form [source_node_index,target_node_index,weight,enable_bit]
        self.test_connection_genes = np.array([[0,3,0.3,1],[0,4,0.4,1],[0,5,0.5,1],[3,2,3.2,1],[4,1,4.1,1],[5,1,5.1,1]])
        #input vector to the NN (one for each input node)
        self.test_input_vector = np.array([1])
        #the hand computed outputs of node 1 and 2
        self.correct_output = np.array([4.19,0.96])

        '''
        Hard code in another test case with recurrent connectionss
        '''
        self.test_node_labels_2 = np.array([-1,1,1,0,0,0])
        #hard code in the connection genes of a simple pre-defined neural network
        #our connection gene convention is of the form [source_node_index,target_node_index,weight,enable_bit]
        self.test_connection_genes_2 = np.array([[0,3,0.3,1],[0,4,0.4,1],[0,5,0.5,1],[3,2,3.2,1],[4,1,4.1,1],[5,1,5.1,1],[3,4,1.5,1],[1,3,1.3,1]])
        #input vector to the NN (one for each input node)
        self.test_input_vector_2 = np.array([1])

    def test(self):
        #creating a test genome 
        test_genome = genome_builder().build_genome(self.test_node_labels,self.test_connection_genes,num_nodes = 8)
        visualise_genome().plot_network(test_genome)
        #find the output of our neural network given our input vector
        output_vector=feedforward().timestep_propagation(test_genome,self.test_input_vector,10e-6,'linear')

        #testing the feedforward_function
        #find the output of our neural network given our input vector

        print(f'The correct output is: {self.correct_output}. The computed output is {output_vector}')
        #if the computed output is (to a very small tolerance) identical to the correct output
        if np.abs(np.sum(self.correct_output-output_vector)) < 10e-10:
            print('Feedforward verified.')
        else:
            raise ValueError(f'Feedforward not verified: Expected {self.correct_output}, got {output_vector}')       
        pass
    
    def test_recurrent(self):
        
        #creating a test genome 
        test_genome = genome_builder().build_genome(self.test_node_labels_2,self.test_connection_genes_2,num_nodes = 8)
        contains_cycle =mutation_functions().contains_cycle
        print(f'True if genome contains a cycle: {contains_cycle(test_genome.connection_enable_bits)}')
        visualise_genome().plot_network(test_genome)
        #create the class which handles functions on this genome
        output_vector = feedforward().timestep_propagation(test_genome,self.test_input_vector,10e-6,'linear')
        print(output_vector)
        pass   
    
