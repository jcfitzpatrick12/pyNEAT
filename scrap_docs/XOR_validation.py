'''
Okay, now onto our XOR validations!

NOTE: WE NEED TO ADD THE MUTATIONS TO THE DICTIONARY!
NOTE: MAKE SURE TO TAKE OUT MANUALLY SET rvs!
NOTE: MAKE SURE NOT TO SET THE RANDOM SEED!
NOTE: Probably rejig the order of operations in assigning innovation number
right now, assigning and then reassigning may be construed as overly complex
'''


import numpy as np
from neat_functions.initialise_networks import InitialiseNetwork
from XOR_verifications.fitness_funcs import FitnessFuncs
from neat_functions.species import species_functions
from neat_functions.reproduction import ReproductionFuncs
from neat_functions.multi_network_mutations import MultiNetworkMutationsFuncs

#np.random.seed(5)
#build the genome collection for the XOR validationg
num_networks = 10
requested_node_labels = np.array([-1,-1,-1,1])
requested_edges = np.array([[0,3],[1,3],[2,3]])
genome_collection = InitialiseNetwork().build_genome_collection(num_networks,requested_node_labels,requested_edges)

#run the feedforward on each genome in turn (for each of the four binary inputs) and compute the network fitness for each genome
genome_collection = FitnessFuncs().XOR_find_network_fitness(genome_collection)

#assign each genome into a new species
genome_collection = species_functions().assign_new_species(genome_collection)

#compute the adjusted fitness for each network!
#we need the network fitness of each genome, the genome that it belongs to and the dictionary which tells us 
#each genomes new species!
genome_collection = FitnessFuncs().XOR_find_adjusted_fitness(genome_collection)
#assign the offspring based on the summed adjusted fitness within each species
genome_collection = ReproductionFuncs().assign_offspring(genome_collection)

#mutate each network in turn, making sure that if the same mutation happens by chance more than once, we assign it the same innovation number
genome_collection = MultiNetworkMutationsFuncs().return_mutated_collection(genome_collection)

