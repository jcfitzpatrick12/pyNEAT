'''
class used to define a genome
convenient for calling node and connection genes
'''

import numpy as np


#genome handles functions related to 
class genome:
    def __init__(self,node_genes,connection_genes):
        self.connection_genes = np.array(connection_genes)
        self.node_genes=np.array(node_genes)
        self.num_total_nodes = len(self.node_genes[:,0])

    #returns the indices of the output neurons
    def find_output_node(self):
        #finding the number of input and output nodes
        num_input_neurons = len(self.node_genes[self.node_genes[:,1]<0])
        num_output_neurons = len(self.node_genes[self.node_genes[:,1]>0])
        #finding the indices of the output nodes (assuming the index ordering [inputs,outputs,hidden_nodes])
        output_node_inds = np.arange(num_input_neurons,num_input_neurons+num_output_neurons)
        #return them
        return output_node_inds
