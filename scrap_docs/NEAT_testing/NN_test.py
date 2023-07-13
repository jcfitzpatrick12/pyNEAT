'''
A test script to see how we might define, implement and forward propagate general neural network topologies

create a function that generates the adjacency matrix 

Not concerned with efficient programming yet!

'''

#import numpy
import numpy as np
#genome class handles functions relating to an individual genome g composed of connection genes and node genes
from genome_functions import genome_functions
#import the class that defines the genome of each network
from genome import genome
#import the testing classes
from test_feedforward_example import test_feedforward_example
from test_mutation_example import test_mutation_example


####################
#END_OF_USER_INPUTS#
####################

#invoking the test handcomputed example
#test_feedforward_example().test()
test_mutation_example().test_single_mutation()












