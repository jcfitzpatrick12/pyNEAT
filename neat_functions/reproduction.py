'''
class which takes in "fully filled" genome_collection (i.e. each genome_descriptor is fully described with genome,fitness and adjusted_fitness!)
and creates a NEW genome_collection full of its offspring according to NEAT reproduction rules.

"Every species is
assigned a potentially different number of offspring in proportion to the sum of ad-
justed fitnesses f ′i of its member organisms. Species then reproduce by first eliminating
the lowest performing members from the population. The entire population is then
replaced by the offspring of the remaining organisms in each species."

So first:

order the genomes by their adjusted fitness within each species
remove a percentage of the lowest performers.

Block into smaller chunks of code and check!

Then onto mutation and keeping track of mutations that have happened in the same generation.

Then... thats it! Try* the XOR validation.
'''

from sys_vars import sys_vars
import numpy as np
from numpy.random import randint
from two_genome_functions.mating import mating
from neat_functions.genome_collection import GenomeCollection

#
class ReproductionFuncs:
    def __init__(self):
        self.sys_vars=sys_vars()

    def find_parent_indices(self,num_networks_in_species):
            #print(num_networks_in_species)
            prob_bounds=np.linspace(0,1,num_networks_in_species+1)
            #recall the parents are already ordered by fitness!
            #use a numpy triangular distribution to infer the parents!
            #ALLOW ASEXUAL REPRODUCTION! In that we may have both parents be identical
            rvs = np.random.triangular(0.25,1,1,size=2)
            #rvs = np.random.uniform(0,1,size=2)
            #rvs = np.array([rv1,rv2])
            # Find the lower bounds for each random variable
            # from this we naturally infer the index of the parent, where the likelihood of choosing a fitter parent is greater!
            parent_indices = np.searchsorted(prob_bounds, rvs)-1
            return parent_indices

    def assign_offspring(self,parent_collection,stagnant_species_dict):
        original_num_networks = parent_collection.return_num_networks()
        parent_collection.reorder_genomes_according_to_adjusted_fitness()
        number_of_offspring_for_each_species_dict = parent_collection.assign_offspring_numbers(original_num_networks,stagnant_species_dict)
        offspring_genomes_dict = {}

        for species_index, parent_descriptors in parent_collection.genomes_dict.items():
            num_networks_in_species = len(parent_descriptors)
            offspring_genomes_dict[species_index] = []
            num_of_champions = int(self.sys_vars.num_of_champions_pc*num_networks_in_species)
            #print('considering new species')
            #print(species_index,len(parent_descriptors))

            # Adjust the number of offspring to generate considering the number of champions
            num_offspring_to_generate = number_of_offspring_for_each_species_dict[species_index]

            if number_of_offspring_for_each_species_dict[species_index] != 0:
                for n in range(num_offspring_to_generate):
                    # If we are adding champions, we need to keep space for them
                    if self.sys_vars.keep_champions and n >= (num_offspring_to_generate - min(num_of_champions, len(parent_descriptors))):
                        break
                    
                    #print(num_offspring_to_generate)
                    chosen_parent_indices = self.find_parent_indices(num_networks_in_species)
                    #print(chosen_parent_indices)
                    chosen_parent0_genome = parent_descriptors[chosen_parent_indices[0]]['genome']
                    chosen_parent1_genome = parent_descriptors[chosen_parent_indices[1]]['genome']
                    chosen_parent0_adjusted_fitness = parent_descriptors[chosen_parent_indices[0]]['adjusted fitness']
                    chosen_parent1_adjusted_fitness = parent_descriptors[chosen_parent_indices[1]]['adjusted fitness']
                    offspring_genome = mating().mate(chosen_parent0_genome,chosen_parent1_genome,chosen_parent0_adjusted_fitness,chosen_parent1_adjusted_fitness)
                    offspring_descriptor = {"genome": offspring_genome, "fitness": None, "adjusted fitness": None}
                    offspring_genomes_dict[species_index].append(offspring_descriptor)

                # If we are adding champions, copy these into the offspring
                if self.sys_vars.keep_champions:
                    for i in range(min(num_of_champions, len(parent_descriptors))):
                        if len(offspring_genomes_dict[species_index]) >= number_of_offspring_for_each_species_dict[species_index]:
                            break  # stop if we've reached the total number of offspring for this species
                        champion_index = len(parent_descriptors) - (i+1)
                        champion_descriptor = parent_descriptors[champion_index]
                        champion_descriptor['fitness'] = None
                        champion_descriptor['adjusted fitness'] = None
                        offspring_genomes_dict[species_index].append(champion_descriptor)
                
        offspring_collection = GenomeCollection(offspring_genomes_dict)
        offspring_total = offspring_collection.return_num_networks()

        if offspring_total != original_num_networks:
            raise ValueError(f'Proposed offspring total not equal to original number of genomes. Expected {original_num_networks}, got {offspring_total}.')
        
        return offspring_collection

