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

import numpy as np
from genome import genome

#given a hard coded neural network, create a genome object (in adjacency form which is our 'default format' of a genome)
class genome_builder:
    def build_genome(self,requested_node_genes,requested_connection_genes,**kwargs):
        #default number of nodes are those which are requested
        default_num_nodes = len(requested_node_genes)
        #however, we can specify a GREATER number of nodes to allow for vector calculations later on
        #this will mean all of our node gene arrays are of the same shape (even if different networks have different numbers of nodes!)
        num_nodes = kwargs.get('num_nodes',default_num_nodes)
        '''
        first build the node genes
        '''
        #we have three entries in the last axis: the index of the node, the nodes labels, and whether that node is enabled
        node_genes = np.zeros((num_nodes,2))
        #label all the requested nodes
        #NOTE all disabled nodes are automatically labelled as hidden by default
        node_genes[:default_num_nodes,0]=requested_node_genes
        #enable all the requested nodes, and keep others disabled
        node_genes[:default_num_nodes,1]=np.ones((default_num_nodes))

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