'''
class to keep track of which babies not allowed to reproduce!
'''
from sys_vars import sys_vars
import numpy as np

class HistoricalRecords:
    def __init__(self):
        self.sys_vars=sys_vars()

    def find_stagnant_species(self, historical_max_fitness_dict):
        # define a dictionary which labels if a species is stagnant or not
        stagnant_species_dict = {}

        # test historical dictionary
        #historical_max_fitness_dict = {0: [0,1,2,3,3,3,3,3,3], 1:[0,1,2,3,4,5],2: [0,1,2,3,3,3]}

        for species_index, historical_max_fitness_list in historical_max_fitness_dict.items():
            #print(f'species index is {species_index}')
            # first assume the species is not stagnant
            stagnant_species_dict[species_index] = False
            # keep track of how many generations this species has been stagnant for
            stagnant_for_this_many_generations = 0
            # store what the max fitness is so far
            max_fitness_so_far = historical_max_fitness_list[0]

            # successfully over each generation, extract the max up to the ith generation
            for i in range(1, len(historical_max_fitness_list)):
                # extract the maximum fitness from this subset
                max_fitness_now = historical_max_fitness_list[i]
                # if the max fitness has improved, assign this to max fitness
                #diff = max_fitness_now - max_fitness_so_far
                if max_fitness_now>max_fitness_so_far+0.1:
                    max_fitness_so_far = max_fitness_now
                    # Reset the stagnation counter as there was improvement
                    stagnant_for_this_many_generations = 0
                else:
                    # otherwise, the species has stagnated
                    stagnant_for_this_many_generations += 1

                if stagnant_for_this_many_generations >= self.sys_vars.stagnant_index-1:
                    # first assume the species is not stagnant
                    stagnant_species_dict[species_index] = True
                    # No need to check further generations for this species as it is already identified as stagnant
                    break 

        return stagnant_species_dict

    
    #keeps track of the max fitnesses of each species after each generation
    def update_dict(self,this_gen_dict,historical_dict,genome_collection):
        #find the max key value of the historical max fitness dict
        for species_index in genome_collection.genomes_dict.keys():
            #print(this_gen_max_fitness_dict[species_index])
            #print(this_gen_max_fitness_dict[species_index])
            try:
                historical_dict[species_index].append(this_gen_dict[species_index])
            except:
                historical_dict[species_index]=[]
                historical_dict[species_index].append(this_gen_dict[species_index])
        return historical_dict