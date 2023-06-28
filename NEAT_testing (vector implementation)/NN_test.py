
'''
Similar to NEAT testing but opt for a vector implementation

-implement input verifications 
'''

import numpy as np
from test_feedforward_example import test_feedforward_example
from test_mutation_example import test_mutation_example
from genome import genome_builder
from visualise_genome import visualise_genome


'''
mutation unit testing for a single network
'''
#test_feedforward_example().test()
test_mutation_example().test_single_mutation()

'''
#perform the feedforward unit test
#test_feedforward_example().test()
#input nodes are labelled by -1, hidden nodes by 0 and output nodes by 1
test_node_genes = np.array([[0,-1],[1,1],[2,1],[3,0],[4,0],[5,0]])
#hard code in the connection genes of a simple pre-defined neural network
#our connection gene convention is of the form [source_node_index,target_node_index,weight,enable_bit,innovation_number]
test_connection_genes = np.array([[0,3,0.3,1,0],[0,4,0.4,1,1],[0,5,0.5,1,2],[3,2,3.2,1,3],[4,1,4.1,1,4],[5,1,5.1,1,5],[1,1,1.1,1,6]])
#creating a test genome 
test_genome = genome_builder().build_genome(test_node_genes,test_connection_genes,)
#plot the test_genome
visualizer = visualise_genome(test_genome)
visualizer.plot_network()
'''
