

#import numpy 
import numpy as np
#import our available activation functions
from activation_functions import activation_functions
#import the class that organises the system variables
from sys_vars import sys_vars
#import the genome class
from genome import genome
#set the random seed
np.random.seed(7)
'''
Define the class which handles operations on a single genome defined by connection_genes and node_genes.
Functions include
-feedforward (which computes the output of a network according to the timestep method)
-mutate (which mutates the genome according to input probabilites)
'''

#genome functions takes an input genome as an input (which is a class object)
class genome_functions:
    def __init__(self,input_genome):
        self.input_genome=input_genome
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
    
    #propagates the inputs through the network defined by connection_genes and node_genes
    #input vector (must be same length of number of input nodes)
    #tol is the pre-defined tolerance to end the timestep method of determining outputs (see docs)
    #requested activation function is a user defined string
    def feedforward(self,input_vector,tol,requested_activation_function):
        #to reduce clutter, extracting the node and connection genese from the input genome
        node_genes= self.input_genome.node_genes
        connection_genes = self.input_genome.connection_genes
        #extracting the points of the connected edges
        target_nodes = connection_genes[:,1]
        #find the output node indices
        #we will use this in the loop
        output_node_inds=self.input_genome.find_output_node()
        #finding the number of total nodes
        num_total_nodes =self.input_genome.num_total_nodes
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
    
    #takes in an (arb)itrary input genome and returns a genome with mutated weights
    def mutate_weights(self,arb_genome):
        #to reduce clutter, extracting the node and connection genese from the input genome
        node_genes= arb_genome.node_genes
        connection_genes = arb_genome.connection_genes
        '''
        first decide whether the weights of this network will be perturbed or not
        '''
        #manually set rv right now to make sure we adjust the weights
        rv = np.random.uniform(low=0.0, high=1.0, size=None)
        #rv=0.4
        #decide whether we will perturb the weights for this network
        if rv>self.sys_vars.probability_weight_perturbed:
            #if we the weights are not perturbed simply pass
            pass
        #otherwise perturb the weights
        else:
            #extract the weights from the connection genes
            weights = connection_genes[:,2]
            #The random variable should be drawn FOR EACH WEIGHT
            rvs = np.random.uniform(low=0.0,high=1.0,size=np.shape(weights))
            #again, for now just manually set the random variables
            #rvs = np.array([0.1,0.91,0.5])
            #create an array of bools which will store whether the weight will be randomly perturbed or not
            boole = np.ones(np.shape(weights))
            #stores one for uniformly perturb, zero for assign a new value
            boole[rvs>self.sys_vars.probability_uniform_weight_perturbation]*=0
            #redefine boole as a boolean array
            boole_weights_uniformly_perturbed=np.array(boole,dtype=bool)
            #create uniform perturbations of ALL WEIGHTS
            uniformly_perturbed_weights = weights + np.random.uniform(-self.sys_vars.uniform_perturbation_radius,self.sys_vars.uniform_perturbation_radius,size=np.shape(weights))
            #create assigned new values of ALL WEIGHTS
            randomly_assigned_new_weights = np.random.uniform(self.sys_vars.weight_range[0],self.sys_vars.weight_range[1],size=np.shape(weights))
            #then simply np.multiply them with their respective boolean arrays and element-wise sum to obtain the result
            #the result being that each weight is now either uniformly perturbed or assigned a new value based on its 
            #drawn random variable
            new_weights = np.multiply(randomly_assigned_new_weights,~boole_weights_uniformly_perturbed)+np.multiply(uniformly_perturbed_weights,boole_weights_uniformly_perturbed)

        #reassign the new_weights
        connection_genes[:,2]=new_weights
        #redefine the genome to have the new weights, and return
        mutated_weights_genome = genome(node_genes,connection_genes)
        return mutated_weights_genome
    
    def mutate(self):
        
        #mutate the weights of the input genome
        mutated_weights_genome = self.mutate_weights(self.input_genome)
        #decide whether we will then mutate and add nodes
        mutate_add_nodes_genome = self.mutate_weights(mutated_weights_genome)
        #print(mutated_weights_genome().connection_genes)
        #self.mutate_add_link(self.genome)
        #self.mutate_add_node(self.genome)
        pass 
    
    '''
    #takes in an (arbitrary) input genome and decides whether to add a node during that mutation
    def mutate_add_node(self,genome):
        #to reduce clutter, extracting the node and connection genese from the input genome
        node_genes= genome.node_genes
        connection_genes = genome.connection_genes

        #draw another single random variable to evaluate whether we will add another hidden node
        #rv = np.random.uniform(low=0,high=1,size=None)
        rv = 0.02
        #print(node_genes)
        if rv<self.sys_vars.probability_add_node:
            #create a new numpy array to hold the new nodes
            new_node_genes = np.empty((self.genome.num_total_nodes+1,2))
            #copy old nodes into new nodes
            new_node_genes[:-1,:]=node_genes
            #add a new HIDDEN node
            new_node_genes[-1,:]=np.array([self.genome.num_total_nodes,0])
            #reassign node_genes to be new_node_genes
            
        print(new_node_genes)
        #for now just return the original genome
        mutated_genome=self.genome
        return mutated_genome
    
    def mutate_add_link(self,):
        pass
        
    '''
    





    