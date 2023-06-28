'''
Class which handles genome functions

Expanded from NEAT testing to include
-conversion from hardcoded connections into adjacency form (given a maximum size with in mind vector operations with other genomes)
'''

import numpy as np

#class which defines a genome. Opposed to NEAT testing, we now use the adjacency form as the fundamental form
class genome:
    def __init__(self, node_genes,connection_genes,):
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
        #set all innovation numbers to be initially nan
        connection_genes[...,2]*=np.nan
        #print(np.shape(connection_genes))
        #for each requested connection_gene, place in the adjacency-like matrix
        for connection_gene in requested_connection_genes:
            #find each directed edge that is requested
            dir_edge = np.array([connection_gene[0],connection_gene[1]],dtype=int)
            #locate the corresponding entry in the adjacency matrix and assign the requested vector
            connection_genes[dir_edge[0],dir_edge[1]] = connection_gene[2:]

        '''
        now we have a problem! As of now, all "disabled" edges have 0 innovation number.
        so find all edges with zero innovation number and change to nan to signify they have not yet recieved an
        innovation number
        
        #extract the enable bits
        enable_bits = connection_genes[...,1]
        #find all edges which are disabled
        disabled_edges = enable_bits==0
        #extract the innovation_numbers
        innov_numbers = connection_genes[...,2]
        #if the enable bit is zero, set the innovation number to nan to signify it has not yet been assigned
        innov_numbers[disabled_edges]=np.nan
        #now place this back into connection_genes
        connection_genes[...,2]=innov_numbers
        '''
        #finally, we can create the genome
        output_genome = genome(node_genes,connection_genes)
        return output_genome
