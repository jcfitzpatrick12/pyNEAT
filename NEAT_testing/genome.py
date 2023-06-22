
import numpy as np

'''
Define the class which handles operations on a single genome defined by connection_genes and node_genes.
Functions include
-feedforward (which computes the output of a network according to the timestep method)
-mutate (which mutates the genome according to input probabilites)


'''
class genome:
    def __init__(self,node_genes,connection_genes,input_vector,tol,requested_activation_function):
        self.node_genes=np.array(node_genes)
        self.connection_genes=np.array(connection_genes)
        self.input_vector=np.array(input_vector)
        self.num_total_nodes = len(self.node_genes[:,0])
        self.tol=tol
        self.requested_activation_function=requested_activation_function

    #returns the indices of the output neurons
    def find_output_node(self):
        #finding the number of input and output nodes
        num_input_neurons = len(self.node_genes[self.node_genes[:,1]<0])
        num_output_neurons = len(self.node_genes[self.node_genes[:,1]>0])
        #finding the indices of the output nodes (assuming the index ordering [inputs,outputs,hidden_nodes])
        output_node_inds = np.arange(num_input_neurons,num_input_neurons+num_output_neurons)
        
        #return them
        return output_node_inds
    

    #given an input of node(s) a, a of arbitrary shape, return the output of that (those) node(s)
    def linear_activatation_function(self,a):
        return a
    
    def exponential_activation_function(self,a):
        return 1/(1+np.exp(-4.9*a))
    
    def activation_function(self,a):
        activation_functions_dict = {'linear':self.linear_activatation_function, 'exponential':self.exponential_activation_function}
        func = activation_functions_dict[self.requested_activation_function]
        return func(a)

    #propagates the inputs through the network defined by connection_genes and node_genes
    def feed_forward(self):
        #to reduce clutter
        node_genes=self.node_genes
        connection_genes = self.connection_genes
        #extracting the points of the connected edges
        target_nodes = connection_genes[:,1]
        #find the output node indices
        #we will use this in the loop
        output_node_inds=self.find_output_node()

        #first check if the input vector is of the same dimension as the input nodes
        num_input_neurons = len(node_genes[node_genes[:,1]<0])
        if len(self.input_vector)!=num_input_neurons:
            raise ValueError(f"Input shape mismatch: Expected {num_input_neurons}, got {len(self.input_vector)}")
        
        #node_states is an array which holds the OUTPUT of nodes in the first column, and the INPUTS of nodes in the
        #second column.
        node_states = np.zeros((self.num_total_nodes,2))
        #initally, we assume the outputs of nodes are 0 for non-input nodes, and the input vector for the input nodes.
        node_states[:num_input_neurons,0]=self.input_vector

        #we use the timestep method which will iterate over 2*N timesteps, where N is the total number of nodes
        #Index iterations by iter
        #The input of the ith node will be the weighted sum of all nodes which connect to it.
        #the NEW output will be the activated INPUT for all nodes (except naturally input nodes which are held constant)
        #as described iterate over 2*N timesteps
        for step in range(2*self.num_total_nodes):
            #iterate over non-input neurons
            for i in range(num_input_neurons,self.num_total_nodes):
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
                node_states[i,1] = self.activation_function(ith_node_input)
                
            #calculate the difference between the two columns
            #if they are pairwise (nearly) identical (to tol tolerance), break the loop
            #force terminate if the OUTPUT vector is nearly identical to the previous timestep to within tol tolerance
            #can adapt so that the network itself stabilises...
            #extract the ouput vectors at step-1 and step
            output_vectors = np.array(node_states[output_node_inds,:])
            
            #check if the output has stabilised
            if np.sum(np.abs(np.diff(output_vectors))) <= self.tol and np.sum(output_vectors)!=0:
                print('Terminated after '+str(step)+' iterations.')
                return output_vectors[:,1]
                break
            #otherwise, swap the columns and iterate again (keeping the inputs always the same)
            node_states[num_input_neurons:,0]=node_states[num_input_neurons:,1]

        return output_vectors[:,0]
    
    #mutates a single network according to the values in the paper
    def mutate(self):
        node_genes = self.node_genes
        connection_genes = self.connection_genes




    