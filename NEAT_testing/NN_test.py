'''
A test script to see how we might define, implement and forward propagate general neural network topologies

create a function that generates the adjacency matrix 

Not concerned with efficient programming yet!

todo:
-"default" settings such as mutation probabilities in a seperate class. W


'''

#import numpy
import numpy as np
#genome class handles functions relating to an individual genome g composed of connection genes and node genes
from genome_functions import genome_functions
#import the class that defines the genome of each network
from genome import genome
#import the test node genome class
from test_genome_example import test_genome_example

'''
feed forward parameters
'''
#tolerance to which we terminate the network
tol=10e-6
#activation function ['linear','exponential']
requested_activation_function = 'linear'


####################
#END_OF_USER_INPUTS#
####################

#invoking the test handcomputed example
test_genome_example().test_feedforward()










