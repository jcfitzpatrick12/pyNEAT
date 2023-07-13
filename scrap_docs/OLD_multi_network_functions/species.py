'''
class which takes in a collection of genomes genome_collection and performs speciesation and mating ect.

Will this code cause an issue if there is a species that dies out?**********
'''

from sys_vars import sys_vars
import numpy as np
from numpy.random import randint,uniform
from multi_network_functions.genome_collection import GenomeCollection
from two_genome_functions.compatability_distance import compatability_distance

#perhaps takes in genome_collection as a parent class?
class species_functions:
    def __init__(self):
        self.sys_vars = sys_vars()
    
    '''
    "The distance measure δ allows us to speciate using a compatibility threshold δt.
    An ordered list of species is maintained. In each generation, genomes are sequentially
    placed into species. Each existing species is represented by a random genome inside
    the species from the previous generation. A given genome g in the current generation is
    placed in the first species in which g is compatible with the representative genome of
    that species. This way, species do not overlap.1 If g is not compatible with any existing
    species, a new species is created with g as its representative"  
    '''





    #function which takes in a genome collection, and assigns it new species according to the NEAT paper
    def assign_new_species(self,genome_collection):

        #extract the representative genome for each species
        representative_genomes = genome_collection.extract_representative_genomes()

        #create an array that will hold the new species labels
        new_species_labels = []
        #create a dynamic variable which tells us the max species index
        #this variable is (naturally) initialised as the current maximum index. If a new species of created, this is incremented.
        dynamic_max_species_index = genome_collection.max_species_index
        #so now we have our list of representative genomes. We can now sequentially place all our genomes into new species.

        #extract the evaluate distance function to be outside the loop
        evaluate_distance = compatability_distance().evaluate_distance

        for i in range(genome_collection.num_networks):
            #extact the ith genome
            genome_g = genome_collection.genomes[i]
            #variable to keep track of whether we have assigned the genome a species
            assigned = False
            #now sequentially go through each species, and evaluate the compatability of the ith genome with the jth species
            for j in range(int(dynamic_max_species_index+1)):
                #extact the jth representative genome
                representative_genome = representative_genomes[j] 
                #evaluate the compatability distance of the ith genome with the jth representative.
                delta = evaluate_distance(genome_g,representative_genome)
                #if the compatability distance is less than the (user defined) threshold, 
                if delta<=self.sys_vars.delta_t:
                    #assign genome g to belong to the jth species
                    new_species_labels.append(j)
                    assigned=True
                    #and break the loop (since we have assigned)
                    break
            #if we have reached the final species, and compatability distance is still not below the threshold
            if not assigned:
                #increment dynamic max species index
                dynamic_max_species_index += 1
                #assign this genome a new species label
                new_species_labels.append(dynamic_max_species_index)
                #and make this genome the representative of the new species
                representative_genomes.append(genome_g)

        #now each genome should be labelled anew, so update the species labels and return genome_collections
        genome_collection.update_species_labels(new_species_labels)
        return genome_collection   


