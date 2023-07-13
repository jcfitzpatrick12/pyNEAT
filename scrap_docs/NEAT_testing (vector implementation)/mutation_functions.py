import numpy as np
#import the class that organises the system variables
from sys_vars import sys_vars
#import the genome class
from genome import genome
#directly draw out the uniform distribution sampling
from numpy.random import uniform,randint
#import the rough visualisation code
from visualise_genome import visualise_genome
#set the random seed
#np.random.seed(2)

#genome functions takes an input genome as an input (which is a class object)
class mutation_functions:
    def __init__(self):
        self.sys_vars = sys_vars()

    def mutate(self,input_genome):
        #visualise the genome before mutation
        visualise_genome().plot_network(input_genome)
        #mutate the weights of the input genome
        mutated_genome = self.mutate_weights(input_genome)
        #visualise the genome after mutation
        visualise_genome().plot_network(mutated_genome)
        #mutate via node addition 
        mutated_genome = self.mutate_add_node(input_genome)
        #mutated_genome = self.mutate_add_node(mutated_genome)
        #visualise the genome after mutation
        visualise_genome().plot_network(mutated_genome)
        #mutate via link addition (no cycles allowed!)
        mutated_genome = self.mutate_add_link(mutated_genome,allow_cycles=False)
        #visualise the genome after mutation
        visualise_genome().plot_network(mutated_genome)
        #mutate via link addition (now cycles allowed!)
        mutated_genome = self.mutate_add_link(mutated_genome,allow_cycles=True)
        #visualise the genome after mutation
        visualise_genome().plot_network(mutated_genome)    
        return mutated_genome
    
    '''
    Depth first search for an arbitrary graph
    '''
    def DFS(self,graph, node, visited, rec_stack):
        # Mark the current node as visited and add it to the recursion stack
        visited[node] = True
        rec_stack[node] = True
        # Get all the neighbors of the current node by finding where in the adjacency matrix the value is 1
        for neighbor in np.where(graph[node] == 1)[0]: 
            # If this node hasn't been visited yet, recursively call DFS function
            if not visited[neighbor]:
                # If the recursive call to DFS finds a cycle, propagate that information up the call stack
                if self.DFS(graph, neighbor, visited, rec_stack):
                    return True
            # If the neighboring node is in the recursion stack, this means we have already started visiting
            # it from the current node, i.e., a cycle exists
            elif rec_stack[neighbor]:
                return True

        # Remove the node from the recursion stack as we finish visiting all its descendants
        rec_stack[node] = False
        return False

    def contains_cycle(self,adj_matrix):
        # The number of nodes in the graph is the dimension of the adjacency matrix
        num_nodes = adj_matrix.shape[0]
        # Initialize visited and recursion stack arrays as False for all nodes
        visited = np.full(num_nodes, False)
        rec_stack = np.full(num_nodes, False)
        # Go through all nodes one by one
        for node in range(num_nodes):
            # If a node has not been visited yet, call DFS on it
            if not visited[node]:
                # If DFS finds a cycle, immediately return True
                if self.DFS(adj_matrix, node, visited, rec_stack):
                    return True

        # If we have gone through all nodes and not found a cycle, return False
        return False

    '''
    to add an edge randomly, making sure there is no cycle, we simply add an edge then determine whether it creates a cycle
    using a Randomized Depth-First Search (DFS). If it creates a cycle, then we select a new edge.
    For simplicity currently, we do "keep track" of which edges have been proposed.
    '''
    def add_link_no_cycles(self,arb_genome):
        #first simply proceed as in the case we we allow cycles
        #add a random edge (which may indeed create a cycle, we don't know yet), the edge we activated, and the total set of permissible edges to activate
        mutated_genome,edge_to_activate,permissible_edges_to_activate = self.add_link_allow_cycles(arb_genome)
        #slice our mutated_genome to make sure we don't have extra nodes
        sliced_genome = mutated_genome.return_enabled_genome()
        proposed_enable_bits = np.copy(sliced_genome.connection_enable_bits)
        #check if we have created a cycle
        cycle_exists = self.contains_cycle(proposed_enable_bits)
        #if we have not created a cycle, we got lucky! return the mutated genome with the added link
        if cycle_exists==False:
            return mutated_genome
        #otherwise our proposed edge creates a cycle... so try again! For efficiency, keep track of our proposed edges
        else:
            #keep track of how many times we have tried to add an edge
            n=0
            #a loose upper bound on how many times we should try to add an edge
            N=len(permissible_edges_to_activate[0,:])
            #otherwise, keep adding edges from permissible edges to activate, until we find one which does not introduce a cycle!
            while cycle_exists == True and n<N:
                    #create an array if the source nodes the permissible edges match with the edge activated
                    sources_match = permissible_edges_to_activate[0,:]==edge_to_activate[0]
                    #create an array if the target nodes the permissible edges match with the edge activated
                    targets_match = permissible_edges_to_activate[1,:]==edge_to_activate[1]
                    #create an array which holds true if both the source and target node match
                    true_if_edge_matches = np.multiply(sources_match,targets_match)
                    #find the index at which this happens
                    edge_ind = np.where(true_if_edge_matches==True)[0]
                    #delete from the list of permissible edges to add
                    permissible_edges_to_activate=np.delete(permissible_edges_to_activate,edge_ind,axis=1)
                    #now add an edge from the remaining set of permissible edges to activate
                    mutated_genome,edge_to_activate = self.add_link_allow_cycles_shortened(arb_genome,permissible_edges_to_activate)
                    #extract the enable_bits matrix of the mutated genome
                    #slice our mutated_genome to make sure we don't have extra nodes
                    sliced_genome = mutated_genome.return_enabled_genome()
                    proposed_enable_bits = np.copy(sliced_genome.connection_enable_bits)
                    #check if we have created a cycle
                    cycle_exists = self.contains_cycle(proposed_enable_bits)
                    #increment the number of times we have added an edge
                    n+=1
            #if we reached our upper bound
            if n>=N:
                print("We have tried all available edges! Returning the original genome.")
                return arb_genome
        return mutated_genome
    
    '''
    a function identical to add_link_allow_cycles, except that we already have our list of permissible edges to activate!
    This way, we can also keep track of which edges we have tried to add from the set of all permissible edges.
    Also, we now don't need to return the permissible edges to activate as this is handled in add_link_no_cycles
    '''

    def add_link_allow_cycles_shortened(self,arb_genome,permissible_edges_to_activate):
        #find the number of permissible edges to activate
        num_permissible_edges_to_activate = len(permissible_edges_to_activate[0,:])
        #if we have no permissible edges to activate!
        if num_permissible_edges_to_activate==0:
            print('No edges can be added! Returning genome unchanged.')
            return arb_genome,None
        #select one to activate at random
        rand_int = randint(0,num_permissible_edges_to_activate)
        #edge_to_activate
        edge_to_activate = permissible_edges_to_activate[:,rand_int]
        #finally activate the edge
        #create a copy of arb_genomes connection genes! 
        connection_genes = np.copy(arb_genome.connection_genes)
        #finally, activate this edge, and assign it a new weight and innovation number
        #find the maximum innovation number
        max_innov = np.max(connection_genes[...,2])
        #assign it a randomly assigned weight from the weight range
        randomly_assigned_new_weight = uniform(self.sys_vars.weight_range[0],self.sys_vars.weight_range[1],size=None)
        #create a new edge connecting the old source node (of the edge split) to our new node of weight one, enable that bit and increment the innovation number by one
        connection_genes[edge_to_activate[0],edge_to_activate[1],:]=[randomly_assigned_new_weight,1,max_innov+1]
        #create the mutated genome and return it
        mutated_genome = genome(arb_genome.node_genes,connection_genes)
        #return the mutated_genome and the edge we activated
        return mutated_genome, edge_to_activate
    
    '''
    In add_link_allow_cycles we have to impose restrictions on what edges are allowed:
    -The target neuron may not be an input.
    '''
    def add_link_allow_cycles(self,arb_genome):
        #add a new conneciton between two previously unconnected enabled nodes (subject to above constraints)
        #so need to find nodes which 1) have no edge connecting them 2) both nodes are enabled.
        #first extract the sliced "enabled" genome (i.e. the non-padded genome which contains only enabled nodes)
        sliced_genome=arb_genome.return_enabled_genome()
        sliced_connection_genes = sliced_genome.connection_genes
        sliced_enable_bits = sliced_connection_genes[...,1]
        #extract ALL disabled edges (including those which will not be permissible to enable)
        disabled_edges=np.array(np.where(sliced_enable_bits==0))
        #find the indices of non_input neurons
        indices_of_input_neurons = np.arange(0,sliced_genome.num_nodes(type='input'))
        #the target indices of disabled edges
        disabled_targets = disabled_edges[1,:]
        #boolean array which is true if that edge has an input as a target
        true_if_target_is_input = np.isin(disabled_targets,indices_of_input_neurons)
        #only allow edges to activate which do not have inputs as a target
        permissible_edges_to_activate = disabled_edges[:,~true_if_target_is_input]
        #find the number of permissible edges to activate
        num_permissible_edges_to_activate = len(permissible_edges_to_activate[0,:])
        #if we have no permissible edges to activate!
        if num_permissible_edges_to_activate==0:
            print('No edges can be added! Returning genome unchanged.')
            return arb_genome
        #select one to activate at random
        rand_int = randint(0,num_permissible_edges_to_activate)
        #edge_to_activate
        edge_to_activate = permissible_edges_to_activate[:,rand_int]
        #finally activate the edge
        #create a copy of arb_genomes connection genes! 
        connection_genes = np.copy(arb_genome.connection_genes)
        #finally, activate this edge, and assign it a new weight and innovation number
        #find the maximum innovation number
        max_innov = np.max(connection_genes[...,2])
        #assign it a randomly assigned weight from the weight range
        randomly_assigned_new_weight = uniform(self.sys_vars.weight_range[0],self.sys_vars.weight_range[1],size=None)
        #create a new edge connecting the old source node (of the edge split) to our new node of weight one, enable that bit and increment the innovation number by one
        connection_genes[edge_to_activate[0],edge_to_activate[1],:]=[randomly_assigned_new_weight,1,max_innov+1]
        #create the mutated genome and return it
        mutated_genome = genome(arb_genome.node_genes,connection_genes)
        #return the mutated_genome, the edge we activated and the permissible edges to activate
        return mutated_genome,edge_to_activate,permissible_edges_to_activate
    

    '''
    "In the add connection
    mutation, a single new connection gene with a random weight is added connecting
    two previously unconnected nodes." - NEAT paper
    This function performs this mutation. NOTE: the new link must connect TWO ENABLED NODES
    if a network has not yet enabled a node, it cannot create a new link to it!
    #one of the "options" allows disabling of recursive links!
    #NOTE the default allows recursive links!
    '''
    def mutate_add_link(self,arb_genome,**kwargs):
        #extracting the correct probability for the respective network size
        network_size = arb_genome.network_size()
        if network_size == 'small_network':
            probability_add_link = self.sys_vars.probability_add_link_small_network
        else: 
            probability_add_link = self.sys_vars.probability_add_link_large_network

        #create a boole that tells us whether we will allow us to add cycles to the network
        default_boole = True
        allow_cycles = kwargs.get('allow_cycles',default_boole)
        #now draw a random variable
        rv = uniform(0,1,size=None)
        rv =0.04
        #if we decide to add a link
        if rv<=probability_add_link:
            #if we are going to allow recursive neural networks
            if allow_cycles==True:
                #find the mutated genome
                mutated_genome = self.add_link_allow_cycles(arb_genome)[0]
                return mutated_genome
            #if we are not going to allow recursive neural networks
            elif allow_cycles==False:
                mutated_genome=self.add_link_no_cycles(arb_genome)
                return mutated_genome
        return arb_genome
    '''
    ""In the add connection
    mutation, a single new connection gene with a random weight is added connecting
    two previously unconnected nodes. In the add node mutation, an existing connection is
    split and the new node placed where the old connection used to be. The old connection
    is disabled and two new connections are added to the genome. The new connection
    leading into the new node receives a weight of 1, and the new connection leading out
    receives the same weight as the old connection. This method of adding nodes was cho-
    sen in order to minimize the initial effect of the mutation."" - NEAT paper

    We will consider only enabled edges can be split

    *****NEED TO BE CAREFUL HOW TO HANDLE INNOVATION NUMBERS!****
    EACH NEW EDGE WILL HAVE AN INCREMENTED INNOVATION NUMBER
    ADAPT LATER TO KEEP TRACK OF INNOVATIONS AT EACH MUTATION
    *************************************************************
    
    This function takes in an (arb)itrary genome and performs the add mutation as described above
    However, we need to be careful with our vector implementation
    if we have unenabled nodes USE them
    if we do not have spare nodes INCREASE THE DIMENSION of our adjacency matrices
    we use this method to minimise the size of our arrays (with in mind when we are implementing the full NEAT algorithm)
    '''
    def mutate_add_node(self,arb_genome):
        #np.random.seed(2)
        #important variable: whether there are any spare nodes. spare_nodes is 0 if there are spare nodes
        zero_if_spare_nodes = np.prod(arb_genome.node_enable_bits)
        #draw a random variable
        rv = uniform(0,1,None)
        #rv=0.001
        #if we decide to add a node
        #choose a link at random to split (uniformly chosen amongst the exisiting links)
        if rv<=self.sys_vars.probability_add_node:
            #if there ARE spare nodes, we don't need to increase dimension of the network
            #so enable one of the unenabled nodes
            if zero_if_spare_nodes==0:
                #if we have some spare nodes, choose a uniformly selected edge amongst those enabled and split it as described in NEAT paper
                mutated_genome = self.add_new_node(arb_genome)
                #return the mutated genome
                return mutated_genome
            #if indeed we have no spare nodes, we need to increase the dimension of connection_genes and node_genes
            #to accomadate the new node
            else:
                #as before except first pad the array to allow for an extra node!
                #num_global_max is equivalent to the number of enabled nodes in this case
                num_global_max_nodes=arb_genome.num_nodes()
                #pad out the genome with an extra node
                arb_genome = arb_genome.return_padded_genome(num_global_max_nodes+1)
                mutated_genome = self.add_new_node(arb_genome)
                #return the mutated genome
                return mutated_genome
        return arb_genome
      
    #add_new_node takes in any arbitrary genome (with at least one available padded column) and adds a new node as per the NEAT paper
    def add_new_node(self,arb_genome):
        #important variable: how many nodes are currently enabled in arb_genome
        num_nodes_enabled = arb_genome.num_nodes(type='enabled')
        #extract the node genes from arb_genome
        node_genes = arb_genome.node_genes
        #enable the first unenabled_node (already pre-labelled as a hidden node)
        node_genes[num_nodes_enabled,:]=[0,1]
        #extract our connection genes
        connection_genes = arb_genome.connection_genes
        #we need to formally disable the new node and assign each corresponding edge to have zero weights (before, the corresponding rows for the new node for the
        #descriptors weight and enable bits are all NaN, while they should be zero if the node is enabled.)
        connection_genes[num_nodes_enabled,:num_nodes_enabled+1,0:2]=0
        connection_genes[:num_nodes_enabled+1,num_nodes_enabled,0:2]=0
        #now find which edges are enabled (disabled edges cannot be split)
        old_enable_bits = arb_genome.connection_enable_bits
        existing_edges = np.array(np.where(old_enable_bits==1))
        #how many existing edges do we have?
        n_existing_edges = np.shape(existing_edges)[1]
        #print(n_existing_edges)
        #select one such edge at random (recall, randint works over the half-open interval [low,high)
        rand_int = randint(0,n_existing_edges)
        #find what edge this corresponds to
        edge_tosplit = existing_edges[:,rand_int]
        #we now need to disable the old edge (convert that enable bit to zero)
        connection_genes[edge_tosplit[0],edge_tosplit[1],1]=0
        #now add the two new connections: (old_source,new_node) with unit weight and incremented innovation number
        #find the maximum innovation number
        max_innov = np.max(connection_genes[...,2])
        #create a new edge connecting the old source node (of the edge split) to our new node of weight one, enable that bit and increment the innovation number by one
        connection_genes[edge_tosplit[0],num_nodes_enabled,:]=[1,1,max_innov+1]
        #find the value of the old weight
        old_weight=connection_genes[edge_tosplit[0],edge_tosplit[1],0]
        #create a new edge whose source is our new node, and whose target is the old node
        #we set it's weight to be the weight of the old edge, enable it and increment the innovation number by 2
        connection_genes[num_nodes_enabled,edge_tosplit[1],:]=[old_weight,1,max_innov+2]
        #build the mutated genome and return
        mutated_genome = genome(node_genes,connection_genes)
        return mutated_genome

    #function that takes an arbitrary genome, and returns a genome with mutated weights
    def mutate_weights(self,arb_genome):
        '''
        first decide whether the weights of this network will be perturbed or not
        '''
        #manually set rv right now to make sure we adjust the weights
        rv = uniform(0.0, 1.0, size=None)
        rv=0.4
        #decide whether we will perturb the weights for this network
        if rv>self.sys_vars.probability_weight_perturbed:
            #if we the weights are not perturbed simply pass
            pass
        #otherwise perturb the weights
        else:
            #make a note of the original global max
            #save_original_global_max = arb_genome.num_nodes()
            #shrink the genome to only include enabled_genomes
            #arb_genome=arb_genome.return_enabled_genome()
            #extracting the weight matrix which assigns the weight to each directed edge
            connection_genes=arb_genome.connection_genes
            connection_weights = connection_genes[...,0]
            connection_enable_bits = connection_genes[...,1]
            #draw a random variable FOR EACH WEIGHT! Even, those of value zero!
            rvs = uniform(0,1,size=np.shape(connection_enable_bits))
            #NOTE: Let us only modifify the weights of enabled edges
            rvs = np.multiply(rvs,connection_enable_bits)
            #print(rvs)
            #set all nan values to zero
            rvs[np.isnan(rvs)]=0
            #hold an array that will tell us which elements have been uniformly perturbed, and which have been assigned new values
            true_if_uniformly_perturb = np.logical_and(rvs > 0, rvs < self.sys_vars.probability_uniform_weight_perturbation)
            true_if_assign_new_values = rvs>=self.sys_vars.probability_uniform_weight_perturbation
            #create uniform perturbations of ALL WEIGHTS
            uniformly_perturbed_weights = connection_weights + uniform(-self.sys_vars.uniform_perturbation_radius,self.sys_vars.uniform_perturbation_radius,size=np.shape(connection_weights))
            #create assigned new values of ALL WEIGHTS
            randomly_assigned_new_weights = uniform(self.sys_vars.weight_range[0],self.sys_vars.weight_range[1],size=np.shape(connection_weights))
            #then simply np.multiply them with their respective boolean arrays and element-wise sum to obtain the result
            #the result being that each weight is now either uniformly perturbed or assigned a new value based on its 
            #drawn random variable
            connection_weights = np.multiply(randomly_assigned_new_weights,true_if_assign_new_values)+np.multiply(uniformly_perturbed_weights,true_if_uniformly_perturb)
            '''
            placing the weights back into a genome
            '''
            #extract the old connection_genes
            connection_genes[...,0]=connection_weights
            #returning a new genome
            return genome(arb_genome.node_genes,connection_genes)

        return arb_genome