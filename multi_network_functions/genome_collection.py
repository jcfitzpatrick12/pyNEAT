'''
class which handles operations on the list of genomes (which contains a python list of genome class objects)

Will this code cause an issue if there is a species that dies out?********** I.e. num_species = np.max(species_labels) CHECK
'''

import numpy as np
from numpy.random import randint

'''
GenomeCollection keeps track of genomes in a generation through dictionaries.
Keys index the species (using 0-indexing)
'''
class GenomeCollection:
    def __init__(self,genomes_dict):
        #extract the list of genomes from genome_collection
        self.genomes_dict = genomes_dict
        #what is the maximum species index? 
        self.max_species_index = len(self.genomes_dict)-1

    #find the number of networks
    def return_num_networks(self):
        # Find out the total number of networks in the genome collection
        return len(self.return_flat_list_genomes())
    
    #find the number of input or output nodes
    def return_num_nodes(self,**kwargs):
        default_type = 'global'
        requested_type = kwargs.get('type',default_type)
        # How many output nodes are in each network of the genome collection (this will be identical so we can take it from the zeroth genome in the list)
        return self.return_flat_list_genomes()[0].num_nodes(type=requested_type)
    
    def return_flat_list_genomes(self):
        #Create a flat list of all genomes
        self.genomes = sum(self.return_species(type='not extinct').values(), [])
        return self.genomes
    #function which returns a dictionary containing either
    #all species, all extinct species, or all species that are not extinct.
    def return_species(self,**kwargs):
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

    #function which can find the number of species of a given type
    #default is 'all'
    def return_num_species(self,**kwargs):
        default_type = 'all'
        requested_type = kwargs.get('type',default_type)
        species_requested = self.return_species(type=requested_type)
        return len(species_requested)

    #returns a dictionary where each key value is the representative genome for that species
    def return_representative_genomes(self,):
        #for each species, select a representative of that species at random (each genome within a species is equally likely to get chosen)
        #the ith element of representative genomes contains the representative genome for the ith species (order preserving)
        representative_genomes = {}
        #iterate through the dictionary
        for k, v in self.genomes_dict.items():
            if v:  # list is not empty
                num_in_kth_species = len(v)
                #select the index of a random genome
                random_index = randint(0, num_in_kth_species)
                #set the representative genome for that species
                representative_genomes[k] = v[random_index]
        #return the dictionary of representative genomes
        return representative_genomes
