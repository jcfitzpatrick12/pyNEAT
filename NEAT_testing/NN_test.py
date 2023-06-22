'''
A test script to see how we might define, implement and forward propagate general neural network topologies

create a function that generates the adjacency matrix 

Not concerned with efficient programming yet!

'''

#import numpy
import numpy as np
#genome class handles functions relating to an individual genome g composed of connection genes and node genes
from genome import genome


'''
Hard code in some networks for verification
'''

#hard code in the node genes
#our node_gene_indexing convention is [bias, input_nodes, output_nodes, hidden_nodes]
#this structure will mean it is e
#input nodes are labelled by -1, hidden nodes by 0 and output nodes by 1
node_genes = [[0,-1],[1,1],[2,0],[3,0]]
#hard code in the connection genes of a simple pre-defined neural network
#our connection gene convention is of the form [source_node_index,target_node_index,weight,enable_bit,innovation_number]
connection_genes = [[0,2,0.2,1,0],[0,3,0.3,1,1],[2,1,2.1,1,2],[3,1,3.1,1,3],[1,1,-0.2,1,3]]
#input vector to the NN (one for each input node)
input_vector = [1]

'''

'''



#tolerance to which we terminate the network
tol=10e-6
#activation function ['linear','exponential']
requested_activation_function = 'linear'


####################
#END_OF_USER_INPUTS#
####################

#genome handles the operations on the network, given the defined node and connection genes.
#recall, tol is our "stability" parameter.
#genome holds functions that compute the 
test_genome = genome(node_genes,connection_genes,input_vector,tol,requested_activation_function)
#find the output of our neural network given our input vector
#output_vector=test_genome.feed_forward()







