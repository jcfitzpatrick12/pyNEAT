'''
class which contains a function which runs the neat algorithm
inputs are:

the number of networks, the initial network topology (node_labels and edges), the fitness function and the fitness threshold!
the fitness threshold controls when the algorithm is stopped (i.e. we have reached a solution)

'''

from neat_functions.initialise_networks import InitialiseNetwork
from neat_functions.species import species_functions
from neat_functions.reproduction import ReproductionFuncs
from neat_functions.multi_network_mutations import MultiNetworkMutationsFuncs
from neat_functions.HMF import HMF
from one_genome_functions.visualise_genome import visualise_genome

import matplotlib.pyplot as plt
import numpy as np

from sys_vars import sys_vars

class run_NEAT:
    def __init__(self,num_networks,requested_node_labels,requested_edges,FitnessFuncs,fitness_threshold):
        self.num_networks = num_networks
        self.requested_node_labels = requested_node_labels
        self.requested_edges = requested_edges
        self.FitnessFuncs = FitnessFuncs
        self.fitness_threshold= fitness_threshold
        self.sys_vars = sys_vars()
        self.historical_max_fitnesses_dict = {}
        self.max_fitnesses = []


    #recursively build new generations until we reach a fitness threshold, or an upper bound on the number of generations!
    def recursive_generation_building(self,genome_collection,**kwargs):
        #naturally, first generation is indexed zero. 
        default_generation_number = 0
        #get the generation number
        generation_number = kwargs.get('generation_number',default_generation_number)

        print(f'Testing generation {generation_number}')
        #run the feedforward on each genome in turn (for each of the four binary inputs) and compute the network fitness for each genome
        genome_collection = self.FitnessFuncs().find_network_fitness(genome_collection)

    
        #check if the max fitness is above the fitness threshold
        proposed_best_genome,max_fitness = genome_collection.return_max_fitness()
        print(f'The current max fitness is {max_fitness}')
        #keep track of max fitnesses
        self.max_fitnesses.append(max_fitness)
        if max_fitness>self.fitness_threshold or generation_number>=self.sys_vars.max_generation:
            print(proposed_best_genome)
            print(max_fitness)

            generations = np.arange(0,generation_number+1)
            plt.ylim(2.5,4.5)
            plt.xlabel('Generation Number')
            plt.ylabel('Maximum Fitness Network')
            plt.axhline(y=4)
            plt.plot(generations,self.max_fitnesses)
            return proposed_best_genome,max_fitness

        #for each species, print how many are in each species
        #for species_index,genome_descriptors in genome_collection.genomes_dict.items():
            #print(species_index,len(genome_descriptors))

        #assign each genome into a new species
        genome_collection = species_functions().assign_new_species(genome_collection)
        #compute the adjusted fitness for each network!
        #we need the network fitness of each genome, the genome that it belongs to and the dictionary which tells us 
        #each genomes new species!
        genome_collection = self.FitnessFuncs().find_adjusted_fitness(genome_collection)

        #keep a track of the max fitnesses of each species through each generation
        #we will need this when assigning offspring 
        this_gen_max_fitness_dict=genome_collection.return_max_fitness_dict()

        #keep track of the max fitness of each species
        self.historical_max_fitnesses_dict = HMF().max_fitness_update_dict(this_gen_max_fitness_dict,self.historical_max_fitnesses_dict,genome_collection)
        #print(self.historical_max_fitnesses_dict)
        #create a dictionary that holds for each species, whether we will instruct it not to reproduce
        stagnant_species_dict = HMF().find_stagnant_species(self.historical_max_fitnesses_dict)

        #assign the offspring based on the summed adjusted fitness within each species
        genome_collection = ReproductionFuncs().assign_offspring(genome_collection,stagnant_species_dict)

        #compute the adjusted fitness for each network!
        #we need the network fitness of each genome, the genome that it belongs to and the dictionary which tells us 
        #mutate each network in turn, making sure that if the same mutation happens by chance more than once, we assign it the same innovation number
        
        #raise SystemExit
        genome_collection = MultiNetworkMutationsFuncs().return_mutated_collection(genome_collection)
        #raise SystemExit
        #increment the generation number
        generation_number+=1
        #call the same function again until we have achieved the max fitness    
        return self.recursive_generation_building(genome_collection,generation_number = generation_number)
        
    def run(self):
        #initialise N networks of teh requested topologies with requested node labels
        intial_genome_collection = InitialiseNetwork().build_genome_collection(self.num_networks,self.requested_node_labels,self.requested_edges)
        #recursively build generations until we reach the threshold fitness for at least one network
        #proposed_best_genome,max_fitness = self.recursive_generation_building(intial_genome_collection)
        return self.recursive_generation_building(intial_genome_collection)

