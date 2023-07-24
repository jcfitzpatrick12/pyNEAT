#import numpy
import numpy as np
#genome class handles functions relating to an individual genome g composed of connection genes and node genes
from one_genome_functions.mutation_functions import mutation_functions
#import genome builder to convert the hard coded edges into the adjacency matrices
from one_genome_functions.genome_builder import genome_builder
#import the rough visualisation code
from one_genome_functions.visualise_genome import visualise_genome

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
        self.test_node_labels = np.array([-1,-1,-1,1])
        #hard code in the connection genes of a simple pre-defined neural network
        #our connection gene convention is of the form [source_node_index,target_node_index,weight,enable_bit,innovation_number]
        self.test_connection_genes = np.array([[0,3,0.3,1],[1,3,1.3,1],[2,3,2.3,1]])

    def test_single_mutation(self):
        #creating a test genome 
        #test_genome = genome_builder().build_genome(self.test_node_labels,self.test_connection_genes,num_nodes=7)
        test_genome = genome_builder().build_genome(self.test_node_labels,self.test_connection_genes)
        visualise_genome().plot_network(test_genome)
        for i in range(3):
            print(test_genome.connection_genes[:,:,i])
        '''
        now mutate the genome for one iteration
        '''
        #initialise a dictionary to keep track of mutatated genes this generation!
        mutated_genes_dict = {}
        #each element of the dict is [source,target,innovation_number]
        #initialise the first mutated gene key (this is a dummy key to save checks later on that a mutation has been added!)
        mutated_genes_dict[0]={'edge_pair':None,'innovation_number':None}
        global_max_innov_number = np.max(test_genome.connection_innov_numbers)
        #test the mutation function
        mutated_genome,mutated_genes_dict,global_max_innov_number = mutation_functions().mutate(test_genome,mutated_genes_dict,global_max_innov_number)
        visualise_genome().plot_network(mutated_genome)
        for i in range(3):
            print(mutated_genome.connection_genes[:,:,i])
        return mutated_genome
