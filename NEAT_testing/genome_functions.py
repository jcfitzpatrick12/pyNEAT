

#import numpy 
import numpy as np
#import our available activation functions
from activation_functions import activation_functions
#import the class that organises the system variables
from system_variables import system_variables

'''
Define the class which handles operations on a single genome defined by connection_genes and node_genes.
Functions include
-feedforward (which computes the output of a network according to the timestep method)
-mutate (which mutates the genome according to input probabilites)
'''

#genome functions takes a defined genome as an input (which is itself a class object)
class genome_functions:
    def __init__(self,genome):
        self.genome=genome
    '''
    feedforward functions
    -defining different activation functions
    -computing the output of a NN given output, requested activation function and termination tolerance
    '''
    #function which returns the requested activation function
    def return_requested_activation_function(self,requested_activation_function):
        activation_func = activation_functions().return_requested_activation_function(requested_activation_function)
        return activation_func
    
    #propagates the inputs through the network defined by connection_genes and node_genes
    #input vector (must be same length of number of input nodes)
    #tol is the pre-defined tolerance to end the timestep method of determining outputs (see docs)
    #requested activation function is a user defined string
    def feedforward(self,input_vector,tol,requested_activation_function):
        #to reduce clutter, extracting the node and connection genese from the input genome
        node_genes= self.genome.node_genes
        connection_genes = self.genome.connection_genes
        #extracting the points of the connected edges
        target_nodes = connection_genes[:,1]
        #find the output node indices
        #we will use this in the loop
        output_node_inds=self.genome.find_output_node()
        #finding the number of total nodes
        num_total_nodes =self.genome.num_total_nodes
        #first check if the input vector is of the same dimension as the input nodes
        num_input_neurons = len(node_genes[node_genes[:,1]<0])
        if len(input_vector)!=num_input_neurons:
            raise ValueError(f"Input shape mismatch: Expected {num_input_neurons}, got {len(input_vector)}")
        #node_states is an array which holds the OUTPUT of nodes in the first column, and the INPUTS of nodes in the
        #second column.
        node_states = np.zeros((num_total_nodes,2))
        #initally, we assume the outputs of nodes are 0 for non-input nodes, and the input vector for the input nodes.
        node_states[:num_input_neurons,0]=input_vector
        #extracting the requested activation function
        activation_func = self.return_requested_activation_function(requested_activation_function)
        #we use the timestep method which will iterate over 2*N timesteps, where N is the total number of nodes
        #Index iterations by iter
        #The input of the ith node will be the weighted sum of all nodes which connect to it.
        #the NEW output will be the activated INPUT for all nodes (except naturally input nodes which are held constant)
        #as described iterate over 2*N timesteps
        for step in range(2*num_total_nodes):
            #iterate over non-input neurons
            for i in range(num_input_neurons,num_total_nodes):
                #find the indices of the connection_genes whose target is the ith neuron
                where_relev_connection_genes = np.where(target_nodes==i)
                #isolate the connection genes whose target is the ith node
                relev_connection_genes = connection_genes[where_relev_connection_genes]
                #isolate the source node (indices) which connects to the ith node
                source_nodes = np.array(relev_connection_genes[:,0],dtype=int)
                #isolate the outputs of each source node
                source_outputs = node_states[source_nodes,0]
                #isolate the weights of each source node
                source_to_ith_node_weights = relev_connection_genes[:,2]
                #isolate the enable bits of each source node
                source_to_ith_node_enable_bits = relev_connection_genes[:,3]
                #modified weights mean if a bit is disabled, it will not contribute to the weighted sum
                modified_weights = np.multiply(source_to_ith_node_weights,source_to_ith_node_enable_bits)
                #calculate the input to the ith node
                ith_node_input=np.dot(source_outputs,modified_weights)
                #the output to the ith node is simply the input, activated
                node_states[i,1] = activation_func(ith_node_input)
                
            #calculate the difference between the two columns
            #if they are pairwise (nearly) identical (to tol tolerance), break the loop
            #force terminate if the OUTPUT vector is nearly identical to the previous timestep to within tol tolerance
            #can adapt so that the network itself stabilises...
            #extract the ouput vectors at step-1 and step
            output_vectors = np.array(node_states[output_node_inds,:])
            
            #check if the output has stabilised
            if np.sum(np.abs(np.diff(output_vectors))) <= tol and np.sum(output_vectors)!=0:
                print('Terminated after '+str(step)+' iterations.')
                return output_vectors[:,1]
            #otherwise, swap the columns and iterate again (keeping the inputs always the same)
            node_states[num_input_neurons:,0]=node_states[num_input_neurons:,1]
        return output_vectors[:,0]


    
    #mutates a single network according to the paper
    #mutation variables are stored in system variables
    #returns a new genome (class object)
    def mutate(self):
        node_genes = self.genome.node_genes
        connection_genes = self.genome.connection_genes




    