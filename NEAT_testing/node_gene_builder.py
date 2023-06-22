
'''
class for building a genome given number of inputs, number of outputs and the directed edges
#define the (directed) edges between the nodes. First element is the (tail) node and the second element is the (point) node
#our indexing convention is [bias, input_nodes, output_nodes, hidden_nodes]
'''

import numpy as np

class node_genes_builder:
    def __init__(self,num_input_nodes,num_output_nodes,num_hidden_nodes):
        self.num_input_nodes = num_input_nodes
        self.num_output_nodes = num_output_nodes
        self.num_hidden_nodes = num_hidden_nodes
        self.num_total_nodes = num_input_nodes + num_output_nodes + num_hidden_nodes

    #creates a numpy array which tells us whether we have an input node (-1), an output node (1), or a hidden node (0)
    #each element is a "node gene" i.e. a tuple with (node_index,node_type) where node_index is a natural number and node_type belongs to [-1,0,1] according to above
    def build_node_genes(self):
        #creates an empty array that will hold the node indices and label
        node_genes = np.zeros((self.num_total_nodes,2))
        #simply a unit-incrementing array to define the node indices
        node_index_array = np.arange(self.num_total_nodes)
        print(node_index_array)
        #indexing the nodes
        node_genes[:,0]=node_index_array
        #label all the input nodes
        node_genes[:self.num_input_nodes,1]=-1
        #label all the output nodes
        node_genes[self.num_input_nodes:self.num_input_nodes+self.num_output_nodes,1]=1
        #label all the hidden nodes
        node_genes[self.num_input_nodes+self.num_output_nodes:-1,1]=0
        #return the node genes
        return node_genes
    

        

      

        
