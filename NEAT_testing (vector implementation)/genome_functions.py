#import numpy 
import numpy as np
#import our available activation functions
from activation_functions import activation_functions
#import the class that organises the system variables
from sys_vars import sys_vars
#import the genome class
from genome import genome
#directly draw out the uniform distribution sampling
from numpy.random import uniform,randint
#set the random seed
np.random.seed(7)

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
    #requested activation function is a user-defined string
    def feedforward(self,input_vector,tol,requested_activation_function):
        #check if the input vector is of the same dimension as the input nodes
        num_input_neurons = self.input_genome.num_nodes(type='input')
        if len(input_vector)!=num_input_neurons:
            raise ValueError(f"Input shape mismatch: Expected {num_input_neurons}, got {len(input_vector)}")
        
        #node_states[...,0] holds the output of nodes at step
        #node_states[...,1] holds the output of nodes at step+1
        #initial value is decided by the input vector
        #node_states holds the node outputs for ALL nodes, even disabled nodes whose state_value will always be zero
        node_states = np.zeros((self.input_genome.num_nodes(type='global_max'),2))
        #we assume so that node_states[...,0] at step=0 (i.e. the initial node outputs) are 0 for non-input nodes, and the input values for the input nodes.
        node_states[:num_input_neurons,0]=input_vector


        '''
        use a vector implementation for the feedforward
        '''

        #create a vectorised array whose first two axis are the same shape as the networks adjacency matrix
        node_states_vect = np.zeros((self.input_genome.num_nodes(type='global_max'),self.input_genome.num_nodes(type='global_max'),2))
        #the input nodes get loaded with the input vector. all other nodes are initially set to be zero.
        node_states_vect[:num_input_neurons,:,0]=input_vector

        '''
        node_states_vect is so defined that node_states_vect[i,:,0] as holding n_global_max identical copies of the output of the ith node
        '''

        #extracting the requested activation function
        activation_func = self.return_requested_activation_function(requested_activation_function)

        #extracting the weight matrix and enable_bit_matrix
        connection_weights = self.input_genome.connection_weights
        connection_enable_bits = self.input_genome.connection_enable_bits
        #making only the enabled_weights non-zero
        #first axis indicates the index of the source node
        #second axis indicates the index of the target node
        #so enabled_weights[i,j] gives the enabled weight of the directed egde of (node_i,node_j)
        enabled_weights = np.multiply(connection_weights,connection_enable_bits)
        #find the output node indices
        #we will use this in the loop
        output_node_inds=self.input_genome.find_output_nodes()

        #we use the timestep method which will iterate over 2*N timesteps, where N is the total number of nodes
        #index iterations by "step"
        #The input of the ith node will be the weighted sum of all nodes which connect to it.
        #the NEW output will be the activated INPUT for all nodes (except naturally input nodes which are held constant)
        #as described iterate over 2*N timesteps
        for step in range(2*self.input_genome.num_nodes(type='enabled')*2):
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
            #take any arbitary row of node_states_vect and extract the output nodes at step+1
            output_vectors = np.array(node_states_vect[output_node_inds,0,1])
            ####-------####
            #calculate the difference between the two columns
            #if they are pairwise (nearly) identical (to tol tolerance), break the loop
            #force terminate if the OUTPUT vector is nearly identical to the previous timestep to within tol tolerance
            #can adapt so that the network itself stabilises...
            #extract the ouput vectors at step-1 and step
            #check if the output has stabilised
            if np.sum(np.abs(np.diff(output_vectors))) <= tol and np.sum(output_vectors)!=0:
                print('Terminated after '+str(step)+' iterations.')
                return output_vectors
            #otherwise, swap the columns and iterate again (keeping the inputs always the same)
            node_states_vect[num_input_neurons:,:,0]=node_states_vect[num_input_neurons:,:,1]
        return output_vectors

    def mutate(self):
        #mutate the weights of the input genome
        #mutated_weights_genome = self.mutate_weights(self.input_genome)
        #mutate via node addition 
        mutated_node_genome = self.mutate_add_node(self.input_genome)
        #self.mutate_add_link(self.genome)

        return mutated_node_genome
    
    '''
    ""In the add connection
    mutation, a single new connection gene with a random weight is added connecting
    two previously unconnected nodes. In the add node mutation, an existing connection is
    split and the new node placed where the old connection used to be. The old connection
    is disabled and two new connections are added to the genome. The new connection
    leading into the new node receives a weight of 1, and the new connection leading out
    receives the same weight as the old connection. This method of adding nodes was cho-
    sen in order to minimize the initial effect of the mutation.""

    *****NEED TO BE CAREFUL HOW TO HANDLE INNOVATION NUMBERS!****
    EACH NEW EDGE WILL HAVE AN INCREMENTED INNOVATION NUMBER
    ADAPT LATER TO KEEP TRACK OF INNOVATIONS AT EACH MUTATION
    *************************************************************

    Create a new function to handle:
    if we have spare nodes USE THEM
    if we do not have spare nodes INCREASE THE DIMENSION
    this will mean we keep our arrays 

    '''
    def mutate_add_node_broken(self,arb_genome):
        #important variable: how many nodes are currently enabled in arb_genome
        num_nodes_enabled = arb_genome.num_nodes(type='enabled')
        #important variable: whether there are any spare nodes. spare_nodes is 0 if there are spare nodes
        spare_nodes = np.prod(arb_genome.node_enable_bits)
        #draw a random variable
        rv = uniform(0,1,None)
        rv=0.001
        #if we decide to add a node
        #choose a link at random to split (uniformly chosen amongst the exisiting links)
        if rv<self.sys_vars.probability_add_node:
            #if there ARE spare nodes, we don't need to increase dimension of the network
            if spare_nodes==0:
                '''
                extend our existing node and connection genes to accomadate a new node
                '''
                #increase the dimension of connection tuple and node tuple
                connection_tup_array =np.array(np.shape(arb_genome.connection_genes))
                #increase the matrix size to num_global_max+1,num_global_max+1
                connection_tup_array[:-1]+=1
                #re-convert into a tuple
                connection_tup = tuple(connection_tup_array)
                #note, for convenience node_tup can be derived from connection tup
                node_tup = connection_tup[1:]
                #and finally create the empty arrays to place our new node and connection genes into
                node_genes = np.zeros(node_tup)
                connection_genes = np.zeros(connection_tup)
                #set initially all innovation numbers to be nan
                connection_genes[...,2]*=np.nan
            '''
            add the new node to node genes.
            NOTE need to note which nodes are ENABLED!
            '''
            #note, these arrays are IDENTICAL to our old arrays, except in the last entry
            node_genes[:-1] = arb_genome.node_genes
            #label all the nodes again, including our newly added node
            node_genes[:,0]=np.arange(len(node_genes))
            #enable the final node
            node_genes[num_nodes_enabled,-1]=1

            '''
            now accomodate a new node in connection_genes
            -an EXISTING edge is split (uniformly randomly sampled)
            -that old edge is DISABLED
            -the edge (old_source,new_node) has unit weight
            -the edge (new_node, old_target) has weight equal to the weight of the edge (old_source,old_target)
            -keep consistent by incrementing our innovation number
            '''

            #place our old connection_genes into the new extended array
            connection_genes[:-1,:-1]=arb_genome.connection_genes
            #first choose which existing edge to split!
            #now find which edges EXIST using arb_genomes.connections_enable_bits
            old_enable_bits = arb_genome.connection_enable_bits
            existing_edges = np.array(np.where(old_enable_bits==1))
            #how many existing edges do we have?
            n_existing_edges = np.shape(existing_edges)[1]
            #select one such edge at random (recall, we need to +1 since randint works over the half-open interval [low,high)
            rand_int = randint(0,n_existing_edges+1)
            #find what edge this corresponds to
            edge_tosplit = existing_edges[:,rand_int]
            #we now need to disable the old edge (convert that enable bit to zero)
            connection_genes[edge_tosplit[0],edge_tosplit[1],1]=0
            #now add the two new connections: (old_source,new_node) with unit weight and incremented innovation number
            #find the maximum innovation number
            max_innov = np.nanmax(connection_genes[...,2])
            connection_genes[edge_tosplit[0],num_nodes_enabled,:]=[1,1,max_innov+1]
            #find the value of the old weight
            old_weight=connection_genes[edge_tosplit[0],edge_tosplit[1],0]
            connection_genes[num_nodes_enabled,edge_tosplit[1],:]=[old_weight,1,max_innov+2]
            mutated_genome = genome(node_genes,connection_genes)
            return mutated_genome
        else:
            pass

        return arb_genome



    #mutates the weights of an arbitrary input genome (single iteration)
    def mutate_weights(self,arb_genome):
        connection_weights = arb_genome.connection_genes
        '''
        first decide whether the weights of this network will be perturbed or not
        '''
        #manually set rv right now to make sure we adjust the weights
        rv = uniform(0.0, 1.0, size=None)
        #rv=0.4
        #decide whether we will perturb the weights for this network
        if rv>self.sys_vars.probability_weight_perturbed:
            #if we the weights are not perturbed simply pass
            pass

        else:
            #extracting the weight matrix which assigns the weight to each directed edge
            connection_weights = arb_genome.connection_weights
            #draw a random variable FOR EACH WEIGHT! Even, those of value zero!
            rvs = uniform(0,1,size=np.shape(connection_weights))
            #set all rvs to zero whose corresponding edge has zero weight
            rvs = np.multiply(rvs,connection_weights)
            #again, for now just manually set the random variables
            #rvs = np.array([0.1,0.91,0.5])
            #create an array of bools which will store whether the weight will be randomly perturbed or not
            boole = np.ones(np.shape(connection_weights))
            #stores one for uniformly perturb, zero for assign a new value
            boole[rvs>self.sys_vars.probability_uniform_weight_perturbation]*=0
            #redefine boole as a boolean array
            boole_weights_uniformly_perturbed=np.array(boole,dtype=bool)
            #create uniform perturbations of ALL WEIGHTS
            uniformly_perturbed_weights = connection_weights + uniform(-self.sys_vars.uniform_perturbation_radius,self.sys_vars.uniform_perturbation_radius,size=np.shape(connection_weights))
            #create assigned new values of ALL WEIGHTS
            randomly_assigned_new_weights = uniform(self.sys_vars.weight_range[0],self.sys_vars.weight_range[1],size=np.shape(connection_weights))
            #then simply np.multiply them with their respective boolean arrays and element-wise sum to obtain the result
            #the result being that each weight is now either uniformly perturbed or assigned a new value based on its 
            #drawn random variable
            connection_weights = np.multiply(randomly_assigned_new_weights,~boole_weights_uniformly_perturbed)+np.multiply(uniformly_perturbed_weights,boole_weights_uniformly_perturbed)

            #extract the old connection genes and update only the weights
            connection_genes = arb_genome.connection_genes
            #update the old weights
            connection_genes[...,0]=connection_weights
            #and thus redefine our new mutated genome.
            #the node genes are unchanged, and we place our connection genes with the updated weights
            mutated_genome = genome(arb_genome.node_genes,connection_genes)
            return mutated_genome

        return arb_genome