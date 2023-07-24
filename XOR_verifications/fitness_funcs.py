

import numpy as np
from sys_vars import sys_vars
from one_genome_functions.feedforward import feedforward
from two_genome_functions.compatability_distance import compatability_distance

class XORFitnessFuncs:
    def __init__(self):
        #extracting sys_vars in case we need it.
        self.sys_vars=sys_vars()
    
    #for simplicity, we adjust the fitness of g, by simply dividing by the number of genomes that reside in the same species of g
    def find_adjusted_fitness(self,genome_collection):
        sharing_sum = compatability_distance().sharing_sum
        #for each species list of genome_descriptors, loop through each genome_descriptor and update the "network output" key with the output vectors
        for genome_descriptors in genome_collection.genomes_dict.values():
                #if that species exists
                if len(genome_descriptors)!=0:
                    #then loop through each genome and compute the adjusted fitness
                    for genome_descriptor in genome_descriptors:
                         g = genome_descriptor['genome']
                         #sh_sum = sharing_sum(g,genome_collection)
                         #print(sh_sum,len(genome_descriptors))
                         #find the adjusted fitness
                         genome_descriptor['adjusted fitness']=genome_descriptor['fitness']/len(genome_descriptors)
        return genome_collection    
    
         
    
    #takes in the four hypothesised network outputs and computes the network fitness
    def evaluate_fitness(self,hypothesised_outputs):
        #print(XOR_network_outputs)
        #defining the correct output for each input vector (according to an XOR gate)
        XOR_true_output_vector = np.array([0,1,1,0])
        #find the distance between the network outputs and the "correct" outputs
        error_distance = np.abs((hypothesised_outputs-XOR_true_output_vector))
        sum_square_error = np.sqrt(np.sum(np.power(error_distance,2)))
        network_fitness=4-sum_square_error
        '''
        #sum the differences for each network
        sum_error_diff = np.sum(error_distance)
        #"... result of this error was subtracted from 4 so that higher fitness would reflect better network structure"
        four_minus_sum_error_diff = 4-sum_error_diff
        #square the distance to obtain the network fitnesses
        network_fitness = np.power(four_minus_sum_error_diff,2)
        #note, the maximum network fitness is 16! since in the ideal case, error_distance is zero, so sum_error_diff is 0, and we get that the max network fitness is 16.      
        '''
        return network_fitness     


    #fill in the output vectors and compute the network fitnesses.
    def find_network_fitness(self,genome_collection):
        #defining all binary input combinations
        #the first input node is a bias node, whose input is ALWAYS set to one.
        XOR_input_vectors = [[1,0,0],[1,0,1],[1,1,0],[1,1,1]]   
        #extract the number of XOR input vectors
        num_XOR_input_vectors = len(XOR_input_vectors)  
        #find the number of input nodes
        num_input_nodes = genome_collection.return_num_nodes(type='input')
        #check the input vectors are of the same length as the number of input nodes
        if len(XOR_input_vectors[0])!=num_input_nodes:
            raise ValueError(f"Input vector shape does not match number of input nodes. Expected {genome_collection.num_input_nodes}, got {len(XOR_input_vectors[0])}")
        #for each species list of genome_descriptors, loop through each genome_descriptor and update the "network output" key with the output vectors
        for genome_descriptors in genome_collection.genomes_dict.values():
                for genome_descriptor in genome_descriptors:
                    #extract the nth genome
                    g = genome_descriptor['genome']
                    #create an array to hold each output_vector
                    output_vectors = np.zeros(num_XOR_input_vectors)
                    #now run through each of the four possible combinations, and perform timestep propagation to find the output
                    for m in range(num_XOR_input_vectors):
                        #extract the mth XOR input vector
                        XOR_input_vector = XOR_input_vectors[m]
                        #print(XOR_input_vector)
                        #propagate the input through the nth 
                        output_vector = feedforward().timestep_propagation(g,XOR_input_vector,10e-6,'exponential')
                        #place the output vector into the network_outputs array
                        output_vectors[m]=output_vector  
                        #output_vectors[:] = [0,1,1,0]
                    network_fitness = self.evaluate_fitness(output_vectors)
                    #print(network_fitness)
                    #print(output_vectors)
                    genome_descriptor['fitness']=network_fitness
        #return (order preserving!) network outputs so that the ith element of XOR network outputs corresponds to the ith genome
        return genome_collection