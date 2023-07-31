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
from neat_functions.historical_records import HistoricalRecords
from general_functions.os_funcs import osFuncs
import numpy as np
import os

from sys_vars import sys_vars

class run_NEAT:
    def __init__(self,num_networks,requested_node_labels,requested_edges,FitnessFuncs,fitness_threshold,keyword):
        self.num_networks = num_networks
        self.requested_node_labels = requested_node_labels
        self.requested_edges = requested_edges
        self.FitnessFuncs = FitnessFuncs
        self.fitness_threshold= fitness_threshold
        self.sys_vars = sys_vars()
        self.keyword=keyword

        #the historical information dictionaries! To keep track of max fitnesses and number of networks at each generation
        self.historical_max_fitnesses_dict = {}
        self.historical_num_networks_dict = {}

    def save_run(self):
        #first pack away the historical records into a class to keep track of them
        historical_records_dict={}
        historical_records_dict['Maximum Fitness']=self.historical_max_fitnesses_dict
        historical_records_dict['Number of Networks']=self.historical_num_networks_dict
        
        osFuncs().save_data("historical_records_dict",historical_records_dict,allow_pickle=True)
        osFuncs().save_data("proposed_best_genome",self.proposed_best_genome,allow_pickle=True)

        '''
        #plot the data!
        plots = plotNEAT(historical_records_dict)
        plots.plot_record_all_species(record_type='Maximum Fitness')
        plots.plot_record_all_species(record_type='Number of Networks')      
        '''
        pass


    #recursively build new generations until we reach a fitness threshold, or an upper bound on the number of generations!
    def recursive_generation_building(self,genome_collection,**kwargs):
        #naturally, first generation is indexed zero. 
        default_generation_number = 0
        #get the generation number
        self.generation_number = kwargs.get('generation_number',default_generation_number)

        print(f'Generation {self.generation_number}')
        #run the feedforward on each genome in turn (for each of the four binary inputs) and compute the network fitness for each genome
        #(i.e. "fill in" the previously None valued network fitnesses in each genome descriptor)
        genome_collection = self.FitnessFuncs().find_network_fitness(genome_collection)
    
        #check if the max fitness is above the fitness threshold
        proposed_best_genome,proposed_best_genome_fitness = genome_collection.return_max_value(self.keyword)
        print(f'The current max {self.keyword} is {proposed_best_genome_fitness}')

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
        this_gen_max_fitness_dict,this_gen_num_networks_dict =genome_collection.return_parameter_dicts()
        #keep track of the max fitness of each species
        self.historical_max_fitnesses_dict = HistoricalRecords().update_dict(this_gen_max_fitness_dict,self.historical_max_fitnesses_dict,genome_collection)
        #keep track of the max fitness of each species
        self.historical_num_networks_dict = HistoricalRecords().update_dict(this_gen_num_networks_dict,self.historical_num_networks_dict,genome_collection)

        #if the proposed best genome is sufficiently fit, finish the algorithm!
        if proposed_best_genome_fitness>self.fitness_threshold or self.generation_number>=self.sys_vars.max_generation:
            #return the overall best genome
            self.proposed_best_genome=proposed_best_genome
            #save the history of evolution
            self.save_run()
            return self.proposed_best_genome

        #print(self.historical_max_fitnesses_dict)
        #create a dictionary that holds for each species, whether we will instruct it not to reproduce
        stagnant_species_dict = HistoricalRecords().find_stagnant_species(self.historical_max_fitnesses_dict)

        #assign the offspring based on the summed adjusted fitness within each species
        genome_collection = ReproductionFuncs().assign_offspring(genome_collection,stagnant_species_dict)

        #compute the adjusted fitness for each network!
        #we need the network fitness of each genome, the genome that it belongs to and the dictionary which tells us 
        #mutate each network in turn, making sure that if the same mutation happens by chance more than once, we assign it the same innovation number
        
        #raise SystemExit
        genome_collection = MultiNetworkMutationsFuncs().return_mutated_collection(genome_collection)
        #raise SystemExit
        #increment the generation number
        self.generation_number+=1
        #call the same function again until we have achieved the max fitness    
        return self.recursive_generation_building(genome_collection,generation_number = self.generation_number)
        
    def run(self):
        #initialise N networks of teh requested topologies with requested node labels
        intial_genome_collection = InitialiseNetwork().build_genome_collection(self.num_networks,self.requested_node_labels,self.requested_edges)
        #recursively build generations until we reach the threshold fitness for at least one network
        #proposed_best_genome,max_fitness = self.recursive_generation_building(intial_genome_collection)
        return self.recursive_generation_building(intial_genome_collection)

