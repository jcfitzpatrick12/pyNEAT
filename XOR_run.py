'''
Okay, now onto our XOR validations!

TH

NOTE: WE NEED TO ASSIGN THE GLOBAL MAXIMUM INNOVATION NUMBER TO NEW GENES, NOT THE LOCAL!
right now, we are assigning the local max innovation number!
Generalise the weight distribution (function in a particular script, rather than every time calling for a particuar distribution)
Make add_link_allow_cycles_shortened redundant as this is a nightmare for debugging.
'''


import numpy as np
from neat_functions.run import run_NEAT
#from XOR_verifications.fitness_funcs import XORFitnessFuncs as FitnessFuncs
from XOR_verifications.fitness_funcs import XORFitnessFuncs as FitnessFuncs
from one_genome_functions.visualise_genome import visualise_genome
from one_genome_functions.feedforward import feedforward

#np.random.seed(5)
#build the genome collction for the XOR validationg
num_networks = 300
requested_node_labels = np.array([-1,-1,-1,1])
requested_edges = np.array([[0,3],[1,3],[2,3]])
fitness_threshold=3.9
#run the NEAT algorithm
proposed_best_genome,max_fitness = run_NEAT(num_networks,requested_node_labels,requested_edges,FitnessFuncs,fitness_threshold).run()
visualise_genome().plot_network(proposed_best_genome)

output_vector = feedforward().timestep_propagation(proposed_best_genome,[1,0,0],10e-6,'exponential')
print(output_vector)
output_vector = feedforward().timestep_propagation(proposed_best_genome,[1,0,1],10e-6,'exponential')
print(output_vector)
output_vector = feedforward().timestep_propagation(proposed_best_genome,[1,1,0],10e-6,'exponential')
print(output_vector)
output_vector = feedforward().timestep_propagation(proposed_best_genome,[1,1,1],10e-6,'exponential')
print(output_vector)
