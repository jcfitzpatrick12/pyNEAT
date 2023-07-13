'''
class which handles operations on the list of genomes (which contains a python list of genome class objects)

Will this code cause an issue if there is a species that dies out?********** I.e. num_species = np.max(species_labels) CHECK
'''

import numpy as np
from numpy.random import randint

class GenomeCollection:
    def __init__(self,genomes,species_labels):
        #extract the list of genomes from genome_collection
        self.genomes = np.array(genomes)
        #how many genomes are in genome collection
        self.num_networks = len(self.genomes)
        #how many output nodes are in each network of genome collection (this will be identical so we can take it from the zeroth genome in the list)
        self.num_output_nodes = self.genomes[0].num_nodes(type='output')
        #how many input nodes are in each network of genome collection (this will be identical so we can take it from the zeroth genome in the list)
        self.num_input_nodes = self.genomes[0].num_nodes(type='input')
        #define the species labels
        self.species_labels = species_labels
        #find the number of species
        self.num_species = np.max(species_labels)+1
        #find the max species index
        self.max_species_index = np.max(species_labels)

    '''
    function which updates the species labels for a particular collection of genomes
    '''

    def update_species_labels(self, new_species_labels):
        self.species_labels = new_species_labels
        self.num_species = np.max(new_species_labels)+1
        self.max_species_index = np.max(new_species_labels)

    '''
    function which seperates the genome into the seperate species according to the species labels.
    returns a list where each element is a genome collection for each species!
    '''
    def separate_into_species(self,):
        #create a list to hold the genome collections, where each genome collection holds genomes which all belong to the same labelled species
        species_lists = []
        #loop through each species
        for i in range(int(self.num_species)):
            #boolean array which holds true if the genome belongs to the ith species
            indices_of_ith_species = np.where(self.species_labels==i)[0]
            #extract those genomes
            genomes_in_ith_species = self.genomes[indices_of_ith_species]
            #create the species label
            num_genomes_in_ith_species = len(genomes_in_ith_species)
            ith_species_labels = np.ones(num_genomes_in_ith_species)*i
            #create a genome collection object with all of these genomes which belong to the same species
            genome_collection_ith_species = GenomeCollection(genomes_in_ith_species,ith_species_labels)
            species_lists.append(genome_collection_ith_species)
            
        #return the list of species
        return species_lists
    
    def extract_representative_genomes(self,):
        #find the number of species in genome_collection
        num_species = int(self.num_species)
        #seperate the genome collection into the different species
        species_list = self.separate_into_species()
        #for each species, select a representative of that species at random (each genome within a species is equally likely to get chosen)
        #the ith element of representative genomes contains the representative genome for the ith species (order preserving)
        representative_genomes = []
        for i in range(num_species):
            #extract the genome_collection of the ith species
            ith_genome_collection = species_list[i]
            #find the number of networks in the ith species
            num_in_ith_species = ith_genome_collection.num_networks
            #select the index of a random genome in that species
            random_index = randint(0,num_in_ith_species)
            #and extract that genome from the ith genome collection
            representative_genome = ith_genome_collection.genomes[random_index]
            #append the representative genome to the list of representative genomes
            representative_genomes.append(representative_genome)
        return representative_genomes