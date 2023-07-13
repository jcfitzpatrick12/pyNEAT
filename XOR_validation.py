'''
Okay, now onto our XOR validations!

The code is suitably ready for a nice vector implementation over all the networks
But start simply by looping.

How do we keep track of the ordering of network outputs and network fitnesses?
I will rewrite the code so that genome_collection.genome_dict is so that each dictionary element is:

species_index: [genome1_descriptor,genome2_descriptor,...,genomeN_descriptor]

where N is the number of genomes in the species labelled by species_index
and genome1_descriptor is itself a dictionary where each dictionary elements are 

"genome": genome class object
"output vector": vector of shape according to number of output nodes
"fitness": genome fitness evaluated using output vector
"adjusted_fitness": the adjusted fitness computed following speciation.

is this convoluted?

I don't think so, we could then see that over the course of one generation, we build each item sequentially,

genome -> output_vector -> genome_fitness (sort into new species using genome compatability, but move the entire descriptor!
this way, we can keep track of each networks fitness and there is no discrepancy in the flattening ordering.) -> genome_adjusted_fitness

and then from genome_adjusted_fitness we build the new genomes 

This


'''

import numpy as np
from multi_network_functions.initialise_networks import InitialiseNetwork
from multi_network_functions.multi_network_feedforward import MultiNetworkFeedforward
from multi_network_functions.network_fitness import fitness_functions
from multi_network_functions.species import species_functions


#np.random.seed(5)
#build the genome collection for the XOR validationg
num_networks = 10
requested_node_labels = np.array([-1,-1,-1,1])
requested_edges = np.array([[0,3],[1,3],[2,3]])
genome_collection = InitialiseNetwork().build_genome_collection(num_networks,requested_node_labels,requested_edges)

#run the feedforward on each genome in turn (for each of the four binary inputs)
#note, network outputs is FLATTENED in the sense that the ith element of network outputs
#corresponds to the ith genome in the FLATTENED genome dictionary
#to be explicit (but not necessarily computationally efficient) we return the ordered genomes in the flattened array
genomes,network_outputs = MultiNetworkFeedforward().XOR_timestep_propagation(genome_collection)

#evaluate the fitness against the "true" output_vectors for the XOR gate
#network_fitnesses is also flattened in the same way!
network_fitnesses = fitness_functions().evaluate_fitness_XOR(network_outputs)


#assign each genome into a new species
speciated_genome_collection = species_functions().assign_new_species(genome_collection)

#compute the adjusted fitness for each network!
#we need the network fitness of each genome, the genome that it belongs to and the dictionary which tells us 
#each genomes new species!
adjusted_network_fitnesses = fitness_functions().evaluate_adjusted_fitness_XOR(genomes,network_fitnesses,speciated_genome_collection)

