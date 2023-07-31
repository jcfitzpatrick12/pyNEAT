'''
class which handles operations on the list of genomes (which contains a python list of genome class objects)

Will this code cause an issue if there is a species that dies out?********** I.e. num_species = np.max(species_labels) CHECK
'''

import numpy as np
from numpy.random import randint
from sys_vars import sys_vars

'''
GenomeCollection keeps track of genomes in a generation through dictionaries.
Keys index the species (using 0-indexing)
'''
class GenomeCollection:
    def __init__(self,genomes_dict):
        #extract the list of genomes from genome_collection
        #genomes_dict is a dictionary, where the keys index the species, and the value is a list of genome_descriptors which belong to that species
        #each genome descriptor contains a genome, its output vector, its network fitness, and its adjusted network fitness
        #if the three latter are not yet defined, they are None valued.
        self.genomes_dict = genomes_dict
        #what is the maximum species index? 
        self.max_species_index = len(self.genomes_dict)-1
        self.sys_vars = sys_vars()

    def return_max_value(self,keyword):
        #initialise the max fitness as zero
        max_fitness = 0
        #find the maximum fitness of a species and that genome descriptor
        for species_index,genome_descriptors in self.genomes_dict.items():
            #loop through each genome descriptor
            for genome_descriptor in genome_descriptors:
                #extract the fitness
                fitness = genome_descriptor[keyword]
                #if it is higher than the max fitness, set that as the max fitness
                #and propose that this genome is the one with the best fitness
                if fitness>max_fitness:
                    max_fitness=fitness
                    proposed_genome = genome_descriptor['genome']

        #after running through     
        return proposed_genome,max_fitness


    def assign_offspring_numbers(self, original_num_networks, stagnant_species_dict):
        num_species = self.return_num_species(type='all')
        adjusted_fitness_sum_dict = self.return_adjusted_fitness_sum_dict()

        # set fitness to nan for stagnant species
        for species_index, is_stagnant in stagnant_species_dict.items():
            if is_stagnant:
                adjusted_fitness_sum_dict[species_index] = np.nan

        # calculate total adjusted fitness excluding nans
        total_adjusted_fitness = sum([v for v in adjusted_fitness_sum_dict.values() if not np.isnan(v)])
        
        # calculate number of offspring for each species
        number_of_offspring_for_each_species_dict = {}
        for species_index, fitness in adjusted_fitness_sum_dict.items():
            if np.isnan(fitness):
                number_of_offspring_for_each_species_dict[species_index] = [0, 0]
                continue

            proportion_of_offspring = (1-is_stagnant)*fitness / total_adjusted_fitness
            proposed_number_of_offspring = int(proportion_of_offspring * original_num_networks)
            remainder = proportion_of_offspring * original_num_networks - proposed_number_of_offspring
            number_of_offspring_for_each_species_dict[species_index] = [proposed_number_of_offspring, remainder]

        # handle cases where total offspring is not equal to original number of networks
        offspring_total = sum([v[0] for v in number_of_offspring_for_each_species_dict.values()])

        sorted_species_by_remainder = sorted(number_of_offspring_for_each_species_dict.items(), key=lambda x: x[1][1], reverse=True)

        while offspring_total != original_num_networks:
            if offspring_total < original_num_networks:
                for species_index, _ in sorted_species_by_remainder:
                    if offspring_total == original_num_networks:
                        break
                    number_of_offspring_for_each_species_dict[species_index][0] += 1
                    offspring_total += 1
            else:
                for species_index, _ in reversed(sorted_species_by_remainder):
                    if offspring_total == original_num_networks:
                        break
                    number_of_offspring_for_each_species_dict[species_index][0] -= 1
                    offspring_total -= 1

        # remove the remainders from the dictionary
        number_of_offspring_for_each_species_dict = {k: v[0] for k, v in number_of_offspring_for_each_species_dict.items()}

        return number_of_offspring_for_each_species_dict

    '''
    def old_assign_offspring_numbers(self, original_num_networks, stagnant_species_dict):
        num_species = self.return_num_species(type='all')
        adjusted_fitness_sum_dict = self.return_adjusted_fitness_sum_dict()

        # set fitness to 0 for stagnant species
        for species_index in stagnant_species_dict.keys():
            if stagnant_species_dict[species_index]:  # if species is stagnant
                adjusted_fitness_sum_dict[species_index] = 0

        total_adjusted_fitness = sum([v for v in adjusted_fitness_sum_dict.values()])
        
        # calculate number of offspring for each species
        number_of_offspring_for_each_species_dict = {}
        for species_index in range(num_species):
            sum_of_adjusted_fitness_within_species = adjusted_fitness_sum_dict[species_index]
            proportion_of_offspring = sum_of_adjusted_fitness_within_species / total_adjusted_fitness
            proposed_number_of_offspring = int(proportion_of_offspring * original_num_networks)
            remainder = proportion_of_offspring * original_num_networks - proposed_number_of_offspring
            number_of_offspring_for_each_species_dict[species_index] = [proposed_number_of_offspring, remainder]

        # handle situation where total offspring is not equal to original number of networks
        offspring_total = sum([v[0] for v in number_of_offspring_for_each_species_dict.values()])

        if offspring_total < original_num_networks:
            # Assign remaining networks
            for i in range(original_num_networks - offspring_total):
                tuple_largest_remainder = max(number_of_offspring_for_each_species_dict.items(), key=lambda x: x[1]) 
                species_index_largest_remainder = tuple_largest_remainder[0]
                number_of_offspring_for_each_species_dict[species_index_largest_remainder][0] += 1
                number_of_offspring_for_each_species_dict[species_index_largest_remainder][1] = 0  # Set remainder to 0

        if offspring_total > original_num_networks:
            # Remove surplus networks
            for i in range(offspring_total - original_num_networks):
                tuple_smallest_remainder = min(number_of_offspring_for_each_species_dict.items(), key=lambda x: x[1]) 
                species_index_smallest_remainder = tuple_smallest_remainder[0]
                number_of_offspring_for_each_species_dict[species_index_smallest_remainder][0] -= 1
                number_of_offspring_for_each_species_dict[species_index_smallest_remainder][1] = 0  # Set remainder to 0

        # verify that the total number of offspring is now equal to the original number of networks
        offspring_total = sum([v[0] for v in number_of_offspring_for_each_species_dict.values()])
        if offspring_total != original_num_networks:
            raise ValueError(f'Proposed offspring total not equal to original number of genomes. Expected {original_num_networks}, got {offspring_total}.')

        # remove the remainders from the dictionary
        number_of_offspring_for_each_species_dict = {k: v[0] for k, v in number_of_offspring_for_each_species_dict.items()}

        return number_of_offspring_for_each_species_dict  
    '''
    
    #note, this fuction assumes that the genomes have already been ordered according to their fitness!
    def remove_poorest_performers(self):
        #self.reorder_genomes_according_to_adjusted_fitness()
        #remove the lowest performing networks in each species
        for species,genome_descriptors in self.genomes_dict.items():
            num_in_species=len(genome_descriptors)
            if num_in_species!=0:#if that species exists
                #find the cut off index, to slice out the lowest performers
                # Determine how many genomes to remove
                num_to_remove = int(self.sys_vars.num_lowest_performs_removed_pc * num_in_species)  
                # Remove the weakest genomes
                self.genomes_dict[species] = genome_descriptors[num_to_remove:]
        pass

    def return_adjusted_fitness_sum_dict(self):
        # Create a new dictionary for the sum of adjusted fitnesses per species
        adjusted_fitness_sum_dict = {}
        # Loop over each species in the genomes_dict
        for species_index, genome_descriptors in self.genomes_dict.items():
            # Initialize the sum for this species to 0
            adjusted_fitness_sum = 0
            # Loop over each genome_descriptor in this species
            for genome_descriptor in genome_descriptors:
                # Add the adjusted_fitness of this genome to the sum
                adjusted_fitness_sum += genome_descriptor["adjusted fitness"]
            # Store the sum of adjusted_fitness for this species in the new dictionary
            adjusted_fitness_sum_dict[species_index] = adjusted_fitness_sum
        return adjusted_fitness_sum_dict

    #return dictionaries with relevent parameters for each species
    #max_fitness,number of networks in that species, proposed best genome... etc.
    def return_parameter_dicts(self):
        # Create a new dictionary for the sum of adjusted fitnesses per species
        max_fitness_dict = {}
        num_networks_dict = {}
        # Loop over each species in the genomes_dict
        for species_index, genome_descriptors in self.genomes_dict.items():
            max_fitness=0
            num_networks_in_species = len(genome_descriptors)
            if len(genome_descriptors)!=0:
            # Initialize the sum for this species to 0
                # Loop over each genome_descriptor in this species
                for genome_descriptor in genome_descriptors:
                    fitness=genome_descriptor['fitness']
                    if fitness>max_fitness:
                        max_fitness=fitness

            max_fitness_dict[species_index]=max_fitness
            num_networks_dict[species_index]=num_networks_in_species
            # Store the sum of adjusted_fitness for this species in the new dictionary
            
        return max_fitness_dict,num_networks_dict

    def reorder_genomes_according_to_adjusted_fitness(self):
        for species_index in self.genomes_dict.keys():
            self.genomes_dict[species_index] = sorted(self.genomes_dict[species_index], key=lambda k: k['adjusted fitness'])
        pass

    #function which returns a dictionary of species of requested type ['all','extinct','not extinct']
    #all species, all extinct species, or all species that are not extinct.
    def return_species_dict(self,**kwargs):
            default_type = 'all'
            requested_type = kwargs.get('type',default_type)
            if requested_type == 'all':
                return self.genomes_dict
            if requested_type == 'extinct':
                #create a dictionary of extinct species
                extinct_species_dict = {k: v for k, v in self.genomes_dict.items() if len(v)==0}
                return extinct_species_dict
            if requested_type == 'not extinct':
                #create a dictionary of extinct species
                not_extinct_species_dict = {k: v for k, v in self.genomes_dict.items() if len(v)!=0}
                return not_extinct_species_dict

    #function which returns a species keys of requested type
    #all species, all extinct species, or all species that are not extinct.
    def return_species_keys(self,**kwargs):
            default_type = 'all'
            requested_type = kwargs.get('type',default_type)
            if requested_type == 'all':
                return list(self.genomes_dict.keys())
            if requested_type == 'extinct':
                #create a dictionary of extinct species
                extinct_species_keys = {k: v for k, v in self.genomes_dict.items() if len(v)==0}.keys()
                return list(extinct_species_keys)
            if requested_type == 'not extinct':
                #create a dictionary of extinct species
                not_extinct_species_keys = {k: v for k, v in self.genomes_dict.items() if len(v)!=0}.keys()
                return list(not_extinct_species_keys)
    

    #find the number of networks
    def return_num_networks(self):
        num_total_genomes = sum(len(v) for v in self.genomes_dict.values())
        return num_total_genomes
    
    #find the number of input or output nodes
    #note, all genomes contain the same number of input and output nodes!
    def return_num_nodes(self,**kwargs):
        default_type = 'global'
        requested_type = kwargs.get('type',default_type)
        #find the first non-empty species in genome_collection
        for v in self.genomes_dict.values():
             if len(v)!=0:
                #extract the first genome_descriptor (technically an arbitrary choice, but the first genome is guaranteed to exist if the species is non-empty)
                sample_genome_descriptor = v[0]
                #extract the genome of the sample genome _descriptor
                sample_genome = sample_genome_descriptor['genome']
                return sample_genome.num_nodes(type=requested_type)

    #function which can find the number of species of a given type
    #default is 'all'
    def return_num_species(self,**kwargs):
        default_type = 'all'
        requested_type = kwargs.get('type',default_type)
        species_requested = self.return_species_keys(type=requested_type)
        return len(species_requested)

    #returns a dictionary where each key value is the representative genome for that species
    def return_representative_genomes(self,):
        #reorder according to adjusted fitness
        #for each species, select a representative of that species at random (each genome within a species is equally likely to get chosen)
        #the ith element of representative genomes contains the representative genome for the ith species (order preserving)
        representative_genomes = {}
        #iterate through the genome_colleciton, k indexes the species
        for k, genome_descriptor in self.genomes_dict.items():
            if len(genome_descriptor)!=0:  # if the species exists
                num_in_kth_species = len(genome_descriptor)
                #select the index of a random genome (from the champions?)
                random_index = randint(0, num_in_kth_species)
                #set the representative genome for that species
                representative_genomes[k] = genome_descriptor[random_index]['genome']
        #return the dictionary of representative genomes
        return representative_genomes
