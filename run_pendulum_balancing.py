'''
Need to check: Why does the base completely stop after each interval? How would we make the base movement continous?


write some code that saves the history of evolution adn the proposed best genome, but in seperate files related by name!
'''


import numpy as np
from neat_functions.run import run_NEAT
from one_pendulum_model.fitness_funcs import PendulumBalancingFitnessFuncs
from sys_vars import sys_vars
from plotting.plot_neat import plotNEAT

#np.random.seed(8)
#build the genome collction for the XOR validationg
total_num_networks = 150
#initial networktopology
requested_node_labels = np.array([-1,-1,-1,1])
requested_edges = np.array([[0,3],[1,3],[2,3]])
fitness_threshold = sys_vars().balance_time

#run the NEAT algorithm
run_NEAT(total_num_networks,requested_node_labels,requested_edges,PendulumBalancingFitnessFuncs,fitness_threshold,'time balanced').run()

#load the most recent data!
plotNEAT().plot_pendulum_run()

