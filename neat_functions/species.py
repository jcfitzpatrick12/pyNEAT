'''
class which takes in a collection of genomes genome_collection and performs speciesation and mating ect.

Will this code cause an issue if there is a species that dies out?**********

Assign fittest species as representative? Or another triangular distribution?
'''

from sys_vars import sys_vars
import numpy as np
from numpy.random import randint,uniform
from neat_functions.genome_collection import GenomeCollection
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
        representative_genomes = genome_collection.return_representative_genomes()
        #create a dynamic variable that updates the max species index
        dynamic_max_species_index = genome_collection.max_species_index
        #so now we have our list of representative genomes. We can now sequentially place all our genomes into new species.
        #extract the evaluate distance function to be outside the loop
        evaluate_distance = compatability_distance().evaluate_distance
        #create a new dictionary to hold the genomes in genome_dict, placed into new species
        #we have to
        speciated_genomes_dict = {}

        #create empty lists in each species that exists in the input genome collection
        for v in genome_collection.genomes_dict:
            speciated_genomes_dict[v]=[]

        #initialise each existing species
        #go through each genome, g, in genomes
        for genome_descriptors in genome_collection.genomes_dict.values():
            for genome_descriptor in genome_descriptors:
                #extact the genome
                g = genome_descriptor['genome']
                #variable to keep track of whether we have assigned the genome a species
                assigned = False
                #now sequentially go through each species, and evaluate the compatability of the ith genome with the jth species
                for species, representative_genome in representative_genomes.items():
                    #evaluate the compatability distance of the ith genome with the jth representative.
                    delta = evaluate_distance(g,representative_genome)
                    #if the compatability distance is less than the (user defined) threshold, 
                    if delta<=self.sys_vars.delta_t:
                        #if species not in new_genomes_dict:
                            #new_genomes_dict[species] = []
                        #assign genome g to belong to the jth species
                        #attach the ENTIRE GENOME DESCRIPTOR to the new speciated genome!
                        #this way we keep track of the network fitnesses!
                        speciated_genomes_dict[species].append(genome_descriptor)
                        assigned=True
                        #and break the loop (since we have assigned)
                        break
                #if we have reached the final species, and compatability distance is still not below the threshold
                if not assigned:
                    #increment dynamic max species index
                    dynamic_max_species_index += 1
                    #assign this genome a new species index
                    speciated_genomes_dict[dynamic_max_species_index] = [genome_descriptor]
                    #select g as the representative genome
                    representative_genomes[dynamic_max_species_index] = g

        #return the new genome collection, now grouped into the appropriate species
        return GenomeCollection(speciated_genomes_dict)



