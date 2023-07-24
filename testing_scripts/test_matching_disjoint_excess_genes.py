import numpy as np
from one_genome_functions.visualise_genome import visualise_genome
from one_genome_functions.genome_builder import genome_builder
from two_genome_functions.two_genome_functions import TwoGenomeFunctions

class test_matching_disjoint_excess_example(TwoGenomeFunctions):
    def __init__(self):
        '''
        Hard code in the parent genes in the NEAT paper (figure 4) for verification
        '''
        #hard code in the node genes
        #our node_gene_indexing ordering convention is [input_nodes, output_nodes, hidden_nodes]
        #input nodes are labelled by -1, hidden nodes by 0 and output nodes by 1
        #so node_labels = [-1,...,-1,,1,...,1,0,...,0]
        #all nodes are assumed enabled
        self.parent0_node_labels = np.array([-1,-1,-1,-1,1,0])
        #hard code in the connection genes of a simple pre-defined neural network
        #our connection gene convention is of the form [source_node_index,target_node_index,weight,enable_bit,innovation_number]
        self.parent0_connection_genes = np.array([[1,4,1.4,1],[2,4,2.4,0],[3,4,3.4,1],[2,5,2.5,1],[5,4,5.4,1],[1,5,1.5,1]])
        #define parent2's node and connection_genes
        self.parent1_node_labels = np.array([-1,-1,-1,-1,1,0,0])
        self.parent1_connection_genes = np.array([[1,4,1.4,1],[2,4,2.4,0],[3,4,3.4,1],[2,5,2.5,1],[5,4,5.4,0],[5,6,5.6,1],[6,4,6.4,1],[3,5,3.5,1],[1,6,1.6,1]])

    def test(self):
        #creating the parent genomes
        parent0 = genome_builder().build_genome(self.parent0_node_labels,self.parent0_connection_genes)
        parent1 = genome_builder().build_genome(self.parent1_node_labels,self.parent1_connection_genes)
        #artificially changing the connection_gene innovation numbers to match the paper
        parent0.connection_innov_numbers[1,5]=8
        parent1.connection_innov_numbers[3,5]=9
        parent1.connection_innov_numbers[1,6]=10
        #visualise the genomes
        visualise_genome().plot_network(parent0)
        visualise_genome().plot_network(parent1)

        #test the disjoint,excess and matching genes
        combined_connection_genes = TwoGenomeFunctions().combine_genome_connection_genes(parent0,parent1)
        disjoint_or_excess_genes = TwoGenomeFunctions().find_disjoint_or_excess_genes(combined_connection_genes)
        matching_genes = TwoGenomeFunctions().find_matching_genes(combined_connection_genes)
        num_disjoint,num_excess = TwoGenomeFunctions().num_disjoint_and_excess(1,disjoint_or_excess_genes,combined_connection_genes)
        #print the results (using the example in the paper!)
        print(disjoint_or_excess_genes[:,:,0])
        print(disjoint_or_excess_genes[:,:,1])
        print(matching_genes)
        print(f'The number of disjoint genes is {num_disjoint}')
        print(f'The number of excess genes is {num_excess}')
        pass
        
