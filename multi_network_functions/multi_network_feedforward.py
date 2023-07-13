

import numpy as np
from sys_vars import sys_vars
from one_genome_functions.feedforward import feedforward

class MultiNetworkFeedforward:
    def __init__(self):
        #extracting sys_vars in case we need it.
        self.sys_vars=sys_vars

    def XOR_timestep_propagation(self,genome_collection):
        #defining all binary input combinations
        #the first input node is a bias node, whose input is ALWAYS set to one.
        XOR_input_vectors = [[1,0,0],[1,0,1],[1,1,0],[1,1,1]]   
        #extract the number of XOR input vectors
        num_XOR_input_vectors = len(XOR_input_vectors)  
        #extract all genomes in genome_collection
        genomes = genome_collection.return_flat_list_genomes()
        #find the number of networks in genome collection
        num_networks = genome_collection.return_num_networks()
        #how many output nodes does our network have (since all networks in genome_collections will have the same number of 
        # output nodes, we can take the zeroth element for convenience)
        num_output_nodes = genome_collection.return_num_nodes(type='output')
        num_input_nodes = genome_collection.return_num_nodes(type='input')
        #build an array that will hold the output vector for each network, and each XOR input vector
        XOR_network_outputs = np.zeros((num_networks,num_XOR_input_vectors,num_output_nodes))
        #for each genome in genome_collection, run the feedforward using each of the inputs.
        for n in range(num_networks):
                #extract the nth genome
                genome = genomes[n]
                #check the input vectors are of the same length as the number of input nodes
                if len(XOR_input_vectors[0])!=num_input_nodes:
                    raise ValueError(f"Input vector shape does not match number of input nodes. Expected {genome_collection.num_input_nodes}, got {len(XOR_input_vectors[0])}")
                #now run through each of the four possible combinations, and perform timestep propagation to find the output
                for m in range(num_XOR_input_vectors):
                    #extract the mth XOR input vector
                    XOR_input_vector = XOR_input_vectors[m]
                    #propagate the input through the nth 
                    output_vector = feedforward().timestep_propagation(genome,XOR_input_vector,10e-6,'exponential')
                    #place the output vector into the network_outputs array
                    XOR_network_outputs[n,m]=output_vector

        #return (order preserving!) network outputs so that the ith element of XOR network outputs corresponds to the ith genome
        return genomes,XOR_network_outputs
  
    '''
    #runs timestep propagation on each network in genome_collection sequentially, and returns the output vector for each network.
    #for XOR validation, we run the feedforward four times for each network, once for each input combination (1,0)
    def XOR_timestep_propagation(self,genome_collection):
        #defining the input vectors for each of the above "true" outputs
        #extract the number of XOR input vectors
        num_XOR_input_vectors = len(XOR_input_vectors)
        #extract the list of genomes from genome_collections
        genomes = genome_collection.genomes
        #find the number of networks in genome collection
        num_networks = genome_collection.num_networks
        #how many output nodes does our network have (since all networks in genome_collections will have the same number of 
        # output nodes, we can take the zeroth element for convenience)
        num_output_nodes = genome_collection.num_output_nodes
        #build an array that will hold the output vector for each network, and each XOR input vector
        XOR_network_outputs = np.zeros((num_networks,num_XOR_input_vectors,num_output_nodes))
        #for each genome in genome_collection, run the feedforward using each of the inputs.
        for n in range(num_networks):
                #extract the nth genome
                genome = genomes[n]
                #check the input vectors are of the same length as the number of input nodes
                if len(XOR_input_vectors[0])!=genome_collection.num_input_nodes:
                    raise ValueError(f"Input vector shape does not match number of input nodes. Expected {genome_collection.num_input_nodes}, got {len(XOR_input_vectors[0])}")
                #now run through each of the four possible combinations, and perform timestep propagation to find the output
                for m in range(num_XOR_input_vectors):
                    #extract the mth XOR input vector
                    XOR_input_vector = XOR_input_vectors[m]
                    #propagate the input through the nth 
                    output_vector = feedforward().timestep_propagation(genome,XOR_input_vector,10e-6,'exponential')
                    #place the output vector into the network_outputs array
                    XOR_network_outputs[n,m]=output_vector

        return XOR_network_outputs
     '''