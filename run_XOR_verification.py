'''
Run the XOR verifications.
'''


import numpy as np
from neat_functions.run import run_NEAT
#from XOR_verifications.fitness_funcs import XORFitnessFuncs as FitnessFuncs
from XOR_verifications.fitness_funcs import XORFitnessFuncs
from plotting.plot_neat import plotNEAT


#np.random.seed(8)
#build the genome collction for the XOR validationg
total_num_networks = 150
requested_node_labels = np.array([-1,-1,-1,1])
requested_edges = np.array([[0,3],[1,3],[2,3]])
#the seconds to balance for
fitness_threshold=1.8

#run the NEAT algorithm
run_NEAT(total_num_networks,requested_node_labels,requested_edges,XORFitnessFuncs,fitness_threshold,'fitness').run()

#load the most recent data!
plotNEAT().test_XOR_verification()



