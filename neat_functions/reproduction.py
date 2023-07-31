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

class ReproductionFuncs:
    def __init__(self):
        self.sys_vars=sys_vars()

    def find_parent_indices(self,num_networks_in_species):
            #print(num_networks_in_species)
            prob_bounds=np.linspace(0,1,num_networks_in_species+1)
            #recall the parents are already ordered by fitness!
            #use a numpy triangular distribution to infer the parents! This puts a bias towards fitter parents (assuming of-course
            # that the networks are order by fitness prior to this function being called)
            #ALLOW ASEXUAL REPRODUCTION! In that we may have both parents be identical
            rvs = np.random.triangular(self.sys_vars.parents_lower_bound_prob,1,1,size=2)
            # Find the lower bounds for each random variable
            # from this we naturally infer the index of the parent, where the likelihood of choosing a fitter parent is greater!
            parent_indices = np.searchsorted(prob_bounds, rvs)-1
            return parent_indices

    #function which takes in a parent collection (genome_collection class object containing a dictionary, where each key is the species index
    # and each value is a list of genome descriptors for that species.) and outputs the offspring collection
    # where the number of offspring is inferred from network fitness!
    def assign_offspring(self,parent_collection,stagnant_species_dict):
        #find the number of networks in the original network
        original_num_networks = parent_collection.return_num_networks()
        #reorder the parent genomes according to fitness within their own genomes
        parent_collection.reorder_genomes_according_to_adjusted_fitness()
        #find the number of offspring for each species
        number_of_offspring_for_each_species_dict = parent_collection.assign_offspring_numbers(original_num_networks,stagnant_species_dict)
        #creating the dictionary to keep hold of all offspring genome_descriptors
        offspring_genomes_dict = {}

        #loop through each species, and extract the list of parent descriptors
        for species_index, parent_descriptors in parent_collection.genomes_dict.items():
            #find the number of networks in that species
            num_networks_in_species = len(parent_descriptors)
            #prime an array to hold the new list of offspring descriptors
            offspring_genomes_dict[species_index] = []
            #compute the number of champions (i.e. genomes we will copy over directly from the parent collection)
            num_of_champions = int(self.sys_vars.num_of_champions_pc*num_networks_in_species)

            #adjust the number of offspring to generate considering the number of champions
            num_offspring_to_generate = number_of_offspring_for_each_species_dict[species_index]

            #if the species is not extinct
            if number_of_offspring_for_each_species_dict[species_index] != 0:
                #loop through each offspring we need to generate
                for n in range(num_offspring_to_generate):
                    # If we are adding champions, we need to keep space for them
                    if self.sys_vars.keep_champions and n >= (num_offspring_to_generate - num_of_champions):
                        break
                    
                    #find the indices of the parent  (i.e. the index of the parent descriptor dictionary in the list of parent_descriptors
                    chosen_parent_indices = self.find_parent_indices(num_networks_in_species)
                    #extract the parents genomes
                    chosen_parent0_genome = parent_descriptors[chosen_parent_indices[0]]['genome']
                    chosen_parent1_genome = parent_descriptors[chosen_parent_indices[1]]['genome']
                    #extract the parents fitness
                    chosen_parent0_adjusted_fitness = parent_descriptors[chosen_parent_indices[0]]['adjusted fitness']
                    chosen_parent1_adjusted_fitness = parent_descriptors[chosen_parent_indices[1]]['adjusted fitness']
                    #produce the offspring genome
                    offspring_genome = mating().mate(chosen_parent0_genome,chosen_parent1_genome,chosen_parent0_adjusted_fitness,chosen_parent1_adjusted_fitness)
                    #create a new descriptor for the offspring
                    offspring_descriptor = {"genome": offspring_genome, "fitness": None, "adjusted fitness": None,"champion": False}
                    #and append this to the list of offspring descriptors for that species
                    offspring_genomes_dict[species_index].append(offspring_descriptor)

                # If we are adding champions, copy these into the offspring
                if self.sys_vars.keep_champions:
                    for i in range(num_of_champions):
                        if len(offspring_genomes_dict[species_index]) >= number_of_offspring_for_each_species_dict[species_index]:
                            break  # stop if we've reached the total number of offspring for this species
                        #find the index of the champion
                        champion_index = len(parent_descriptors) - (i+1)
                        #extract the full descriptor of that champion
                        champion_descriptor = parent_descriptors[champion_index]
                        #impose the fitness and adjusted fitness to be zero (for clarity, since the above line takes the non-zero fitness and adjusted fitness from the previous generation)
                        champion_descriptor['fitness'] = None
                        champion_descriptor['adjusted fitness'] = None
                        champion_descriptor['champion']=True
                        #as before, append this to the list of offspring descriptors
                        offspring_genomes_dict[species_index].append(champion_descriptor)
        
        #turn the offspring genome dictionary into the genome_collection class object
        offspring_collection = GenomeCollection(offspring_genomes_dict)
        #find the total number of offspring
        offspring_total = offspring_collection.return_num_networks()

        #this should be identical to the original number of networks!
        if offspring_total != original_num_networks:
            raise ValueError(f'Proposed offspring total not equal to original number of genomes. Expected {original_num_networks}, got {offspring_total}.')
        
        #if it is.... return good as new!
        return offspring_collection

