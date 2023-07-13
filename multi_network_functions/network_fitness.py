'''
class which handles functions that deal with evaluating network fitness

"To compute fitness, the distance of the output from the correct answer was
summed for all four input patterns. The result of this error was subtracted from 4
so that higher fitness would reflect better network structure. The resulting number was
squared to give proportionally more fitness the closer a network was to a solution." - NEAT paper
'''
import numpy as np
from sys_vars import sys_vars

class fitness_functions:
    def __init__(self):
        self.sys_vars=sys_vars()

    #function which takes in the XOR network outputs for each network in genome collections (naturally preserving order so that the nth element
    # of XOR_network_outputs corresponds to the outputs of the nth genome upon timestep propagating the 4 combinations of binary inputs)
    def evaluate_fitness_XOR(self,XOR_network_outputs):
        #print(XOR_network_outputs)
        #defining the correct output for each input vector (according to an XOR gate)
        XOR_true_output_vector = np.array([0,1,1,0])
        #finding the number of genomes we are considering
        num_networks = np.shape(XOR_network_outputs)[0]
        #hstack along the last axis
        XOR_network_outputs = np.hstack(XOR_network_outputs)
        #swap the axis to preserve the structure of XOR_network_outputs
        XOR_network_outputs = np.swapaxes(XOR_network_outputs,axis1=0,axis2=1)
        #vectorise the "correct" outputs so to permit vector calculationss
        XOR_true_output_vector_vectorised = np.repeat(XOR_true_output_vector[:,np.newaxis],num_networks,axis=1)
        XOR_true_output_vector_vectorised = XOR_true_output_vector_vectorised.T
        #find the distance between the network outputs and the "correct" outputs
        error_distance = np.abs((XOR_true_output_vector_vectorised-XOR_network_outputs))
        #sum the differences for each network
        sum_error_diff = np.sum(error_distance,axis=-1)
        #"... result of this error was subtracted from 4 so that higher fitness would reflect better network structure"
        four_minus_sum_error_diff = 4-sum_error_diff
        #square the distance to obtain the network fitnesses
        network_fitnesses = np.power(four_minus_sum_error_diff,2)
        #note, the maximum network fitness is 16! since in the ideal case, error_distance is zero, so sum_error_diff is 0, and we get that the max network fitness is 16.
        return network_fitnesses

    def evaluate_adjusted_fitness_XOR(self,genomes,network_fitnesses,speciated_genome_collection):
        print('here!')
        return None