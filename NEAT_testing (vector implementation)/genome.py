'''
Class which handles genome functions

Expanded from NEAT testing to include
-conversion from hardcoded connections into adjacency form (given a maximum size with in mind vector operations with other genomes)
'''

import numpy as np
from sys_vars import sys_vars
import warnings

#class which defines a genome. Opposed to NEAT testing, we now use the adjacency form as the fundamental form
class genome:
    def __init__(self, node_genes,connection_genes,):
        #make available the system parameters
        self.sys_vars=sys_vars()
        #return the connection genes as a whoe
        self.connection_genes = connection_genes
        #return the adjacency matrix where each entry is the weight of that directed edge
        self.connection_weights = self.connection_genes[:,:,0]
        #return the adjacency matrix where each entry is whether that edge is enabled
        self.connection_enable_bits = self.connection_genes[:,:,1]  
        #this is identically the adjacency matrix
        self.adjacency_matrix = self.connection_enable_bits
        #return the adjacency matrix where each entry is the innovation number of that edge 
        self.connection_innov_numbers = self.connection_genes[:,:,2]

        #return the 'standard' adjacency matrix
        #self.adjacency_matrix = self.connection_weights/self.connection_weights
        #returns the node genese as a whole
        self.node_genes = node_genes
        #returns the index of each node
        self.node_indices = self.node_genes[:,0]
        #returns the label of each node (at that index) input:-1, hidden:0,output:1
        self.node_labels = self.node_genes[:,1]
        #returns whether that node is enabled
        self.node_enable_bits = self.node_genes[:,2]

    #returns the indices of the output neurons
    def find_output_nodes(self):
        #finding the indices of the output nodes (assuming the index ordering [inputs,outputs,hidden_nodes]) and returning them
        return np.arange(self.num_nodes(type='input'),self.num_nodes(type='input')+self.num_nodes(type='output'))
    
    #simple function that returns the size of the network
    def network_size(self,):
        #find the number of ENABLED nodes
        num_nodes = self.num_nodes(type='enabled')
        #if num_nodes is smaller than our defined "small network" size
        if num_nodes <=self.sys_vars.small_network_size:
            return "small_network"
        #if network is not small, it is large
        else:
            return "large_network"

    '''
    now will need functions to extract:
    -number of input nodes
    -number of output nodes
    -number of "enabled" nodes
    '''

    #one function to find all nodes
    #default type is global max (sum of enabled and disabled nodes)
    def num_nodes(self,**kwargs):

        default_type = 'global_max'
        requested_num_nodes = kwargs.get('type',default_type)

        if requested_num_nodes == 'global_max':
            num_nodes = len(self.node_genes[:,0])
            return num_nodes
    
        #if we want to find 'input' nodes
        if requested_num_nodes == 'input':
            #boolean array which holds one if node labels are less than 0
            input_nodes_boole = self.node_labels<0
            #number of input nodes is thus the sum of the above boolean array
            num_nodes = np.sum(input_nodes_boole)
            return num_nodes
        
        #if we want to find 'input' nodes
        if requested_num_nodes == 'output':
            #boolean array which holds one if node labels are more than 0
            output_nodes_boole = self.node_labels>0
            #number of output nodes is thus the sum of the above boolean array
            num_nodes = np.sum(output_nodes_boole)
            return num_nodes
        
        #the number of hidden (ENABLED!) nodes
        if requested_num_nodes == 'hidden':
            #boolean array which holds one if node labels are exactly 0
            output_nodes_boole = self.node_labels==0
            #number of hiddem nodes is thus the difference of all nodes with 0 label the sum of the above boolean array
            num_nodes = len(self.node_genes[:,0])-np.sum(output_nodes_boole)
            return num_nodes
        

        #if we want to find the number of 'enabled' nodes
        if requested_num_nodes == 'enabled':
            #return how many are enabled (if they are enabled, the enable bit will be one. else it will be zero. so simply return the sum of the enable bits)
            num_nodes = int(np.sum(self.node_enable_bits))
            return num_nodes

    #function which returns a "sliced" network which includes only the enabled nodes
    def return_enabled_genome(self):
        #find the number of enabled nodes
        num_enabled_nodes=self.num_nodes(type='enabled')
        #find the global max number of nodes
        num_global_max_nodes =self.num_nodes(type='global_max')
        '''
        #checking if there are any spare nodes at all raise
        if num_enabled_nodes==num_global_max_nodes:
            warnings.warn("No spare nodes to slice. Returning original genome")
            #return an unchanged genome
            return genome(self.node_genes,self.connection_genes)
        '''
        #slice the spare node genes
        node_genes = self.node_genes[:num_enabled_nodes,...]
        #slice the spare connection genes
        connection_genes=self.connection_genes[:num_enabled_nodes,:num_enabled_nodes]
        #build the sliced genome
        sliced_genome = genome(node_genes,connection_genes)
        return sliced_genome
    
    #function_which returns a "padded" network with number of nodes equal to num_nodes
    #all "extra" nodes are set to be disabled
    def return_padded_genome(self,num_nodes):
        #find the number of enabled nodes
        num_global_max_nodes=self.num_nodes(type='global_max')
        #output dependent on requested number of nodes
        if num_nodes<num_global_max_nodes:
            raise ValueError(f"Requested number of nodes must be equal than or greater than the current number of nodes: Expected minimum {num_global_max_nodes}, got {num_nodes}")
        #otherwise pad the extra columns with nans (if we request the same number of nodes, the below code trivially copies over to another array)
        else:
            '''
            extend our existing node and connection genes to accomadate a new node
            '''
            #increase the dimension of connection tuple and node tuple
            connection_tup_array =np.array(np.shape(self.connection_genes))
            #increase the matrix size to num_nodes x requested_global max
            increase_dim = num_nodes-num_global_max_nodes
            connection_tup_array[:-1]+=increase_dim
            #re-convert into a tuple
            connection_tup = tuple(connection_tup_array)
            #note, for convenience node_tup can be derived from connection tup
            node_tup = connection_tup[1:]
            '''
            creating the padded_connection_genes
            '''
            #first axis denotes the source node, second axis denotes the target node and the 3 entries on the last axis are defined above
            padded_connection_genes = np.zeros((connection_tup))
            #set ALL descriptors of connection_genes outwith default_num_nodes to be initially nan valued
            padded_connection_genes[num_global_max_nodes:,:]*=np.nan
            padded_connection_genes[:,num_global_max_nodes:]*=np.nan
            #set ALL innovation_numbers to be initially nan valued
            padded_connection_genes[:,:,2]*=np.nan
            #copy over the old genome
            padded_connection_genes[:num_global_max_nodes,:num_global_max_nodes]=self.connection_genes
            '''
            creating the padded node_genes
            '''
            padded_node_genes = np.zeros((node_tup))
            #label each node
            padded_node_genes[:,0] = np.arange(num_nodes)
            #place in the old node genes
            padded_node_genes[:num_global_max_nodes]=self.node_genes

            #build the sliced genome
            padded_genome = genome(padded_node_genes,padded_connection_genes)
            return padded_genome




        
        

        





'''
Important (rough) notes on connection gene and node gene values
an enabled node has enable bit 1
a disabled node has enable bit 0
this will allow us to consider that all networks are "of the same shape" even if they have a differing number of nodes

the connection genes whose target or source is a disabled node has all descriptors (weight, enable bit, innovation) to be nan
the enable bit in this case is set to nan to distinguish between nodes which "exist" for a particular network
for example in the add link mutation, we need to be able to tell which linkes we are "allowed" to connect to each other!

the connection genes whose target and source are enabled has non-nan weight and enable bits
# if the node is enabled and the edge is not yet enabled it may be enabled in mutate_add_link!
#however, unless that edge is specifically asked for, it will have nan innovation number
#this is so that the "not yet specified" edges don't also have zero innovation number (only one edge may have zero innovation number for a single network)



'''

#given a hard coded neural network, create a genome object (in adjacency form which is our 'default format' of a genome)
class genome_builder:
    def build_genome(self,requested_node_genes,requested_connection_genes,**kwargs):
        #default number of nodes are those which are requested
        default_num_nodes = len(requested_node_genes[:,0])
        #however, we can specify a GREATER number of nodes to allow for vector calculations later on
        #this will mean all of our node gene arrays are of the same shape (even if different networks have different numbers of nodes!)
        num_nodes = kwargs.get('num_nodes',default_num_nodes)
        '''
        first build the node genes
        '''
        #we have three entries in the last axis: the index of the node, the nodes labels, and whether that node is enabled
        node_genes = np.zeros((num_nodes,3))
        #label each node
        node_genes[:,0] = np.arange(num_nodes)
        #label all the requested nodes
        #NOTE all disabled nodes are automatically labelled as hidden by default
        node_genes[:default_num_nodes,1]=requested_node_genes[:,1]
        #enable all the requested nodes, and keep others disabled
        node_genes[:default_num_nodes,2]=np.ones((default_num_nodes))

        '''
        now build the connection genes
        -define an adjaceency-like matrix of the network
        -which (equivalently to below) assigns each entry to vector which contains
        -the weight of that edge (zero if it does not exist)
        -enable bit
        -innovation number
        '''
        #first axis denotes the source node, second axis denotes the target node and the 3 entries on the last axis are defined above
        connection_genes = np.zeros((num_nodes,num_nodes,3))
        #set ALL descriptors of connection_genes outwith default_num_nodes to be initially nan valued
        connection_genes[default_num_nodes:,:]*=np.nan
        connection_genes[:,default_num_nodes:]*=np.nan
        #set ALL innovation_numbers to be initially nan valued
        connection_genes[:,:,2]*=np.nan
        #print(np.shape(connection_genes))
        innov_no = 0
        #for each requested connection_gene, place in the adjacency-like matrix
        for connection_gene in requested_connection_genes:
            #find each directed edge that is requested
            dir_edge = np.array([connection_gene[0],connection_gene[1]],dtype=int)
            #locate the corresponding entry in the adjacency matrix and assign the requested vector
            connection_genes[dir_edge[0],dir_edge[1],:-1] = connection_gene[2:]
            #assign the innovation_number
            connection_genes[dir_edge[0],dir_edge[1],-1]=innov_no
            #increment the innovation number
            innov_no+=1

        #finally, we can create the genome
        output_genome = genome(node_genes,connection_genes)
        return output_genome
