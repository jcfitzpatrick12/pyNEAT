import numpy as np
from genome import genome
from sys_vars import sys_vars
#import our available activation functions
from activation_functions import activation_functions

#genome functions takes an input genome as an input (which is a class object)
class feedforward:
    def __init__(self):
        self.sys_vars = sys_vars()
        
        '''
    feedforward functions
    -defining different activation functions
    -computing the output of a NN given output, requested activation function and termination tolerance
    '''

    #function which returns the requested activation function
    def return_requested_activation_function(self,requested_activation_function):
        activation_func = activation_functions().return_requested_activation_function(requested_activation_function)
        return activation_func
    
    #propagates the inputs through the network defined by connection_genes and node_genes using the timestep method
    #input_genome
    #input vector (must be same length of number of input nodes)
    #tol is the pre-defined tolerance to end the timestep method of determining outputs (see docs)
    #requested activation function is a user-defined string
    def timestep_propagation(self,input_genome,input_vector,tol,requested_activation_function):
        #check if the input vector is of the same dimension as the input nodes
        num_input_neurons = input_genome.num_nodes(type='input')
        if len(input_vector)!=num_input_neurons:
            raise ValueError(f"Input shape mismatch: Expected {num_input_neurons}, got {len(input_vector)}")
        
        #node_states[...,0] holds the output of nodes at step
        #node_states[...,1] holds the output of nodes at step+1
        #initial value is decided by the input vector
        #node_states holds the node outputs for ALL nodes, even disabled nodes whose state_value will always be zero
        node_states = np.zeros((input_genome.num_nodes(type='global_max'),2))
        #we assume so that node_states[...,0] at step=0 (i.e. the initial node outputs) are 0 for non-input nodes, and the input values for the input nodes.
        node_states[:num_input_neurons,0]=input_vector
        '''
        use a vector implementation for the feedforward
        '''
        #create a vectorised array whose first two axis are the same shape as the networks adjacency matrix
        node_states_vect = np.zeros((input_genome.num_nodes(type='global_max'),input_genome.num_nodes(type='global_max'),2))
        #the input nodes get loaded with the input vector. all other nodes are initially set to be zero.
        node_states_vect[:num_input_neurons,:,0]=np.vstack(input_vector)
        '''
        node_states_vect is so defined that node_states_vect[i,:,0] as holding n_global_max identical copies of the output of the ith node
        '''
        #extracting the requested activation function
        activation_func = self.return_requested_activation_function(requested_activation_function)
        #extracting the weight matrix and enable_bit_matrix
        connection_weights = input_genome.connection_weights
        connection_enable_bits = input_genome.connection_enable_bits
        #making only the enabled_weights non-zero
        #first axis indicates the index of the source node
        #second axis indicates the index of the target node
        #so enabled_weights[i,j] gives the enabled weight of the directed egde of (node_i,node_j)
        enabled_weights = np.multiply(connection_weights,connection_enable_bits)
        #find the output node indices
        #we will use this in the loop
        output_node_inds=input_genome.find_output_nodes()

        #we use the timestep method which will iterate over 2*N timesteps, where N is the total number of nodes
        #index iterations by "step"
        #The input of the ith node will be the weighted sum of all nodes which connect to it.
        #the NEW output will be the activated INPUT for all nodes (except naturally input nodes which are held constant)
        #as described iterate over 2*N timesteps
        for step in range(2*input_genome.num_nodes(type='enabled')*2):
            #print(step)
            #recall we assume every node is assumed connected to every other node.
            #if that edge does not belong to the network, it holds zero weight
            #so the i,jth element of edge_weighted_output holds the edge_weighted output of the directed edge (node_i,node_j) 
            #where the first element is the source node, second is the target node
            edge_weighted_output = np.multiply(node_states_vect[...,0],enabled_weights)
            #we now look to find the weighted sum of all outputs whose target is the ith node
            #we define new_node_inputs, whose ith element of holds the weighted sum of all nodes
            #whose TARGET is the ith node. so sum over the zeroth axis!
            new_node_inputs = np.nansum(edge_weighted_output,axis=0)
            #axis=0 means the ith element of new_node_inputs is sum_j weight_prod_node_output[j,i]
            #so we now have the input for each node! The new outputs are simply the inputs to each node activated by the node.
            new_node_outputs = activation_func(new_node_inputs)
            #place in the new node outputs to node_states_vect (recall, the ith row holds n_global_max copies of the output of the ith node at either step or step+1 depending on the last axis)
            node_states_vect[...,1]=np.vstack(new_node_outputs)
            #take any arbitary row of node_states_vect and extract the output nodes at step and step+1
            output_vectors = np.array(node_states_vect[output_node_inds,0,:])
            ####-------####
            #calculate the difference between the two columns
            #if they are pairwise (nearly) identical (to tol tolerance), break the loop
            #force terminate if the OUTPUT vector is nearly identical to the previous timestep to within tol tolerance
            #can adapt so that the network itself stabilises...
            #extract the ouput vectors at step-1 and step
            #check if the output has stabilised
            if np.sum(np.abs(np.diff(output_vectors,axis=-1))) <= tol and np.sum(output_vectors)!=0:
                print('Terminated after '+str(step)+' iterations.')
                return output_vectors[...,1]
            #otherwise, swap the columns and iterate again (keeping the inputs always the same)
            node_states_vect[num_input_neurons:,:,0]=node_states_vect[num_input_neurons:,:,1]
        return output_vectors[...,1]

