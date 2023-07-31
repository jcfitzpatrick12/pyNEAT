'''
class which takes in a genome collection and mutates each network in turn
if the same mutation happens by chance in the same generation, assign it the same innovation number
'''

#from sys_vars import sys_vars
import numpy as np
from one_genome_functions.mutation_functions import mutation_functions
from sys_vars import sys_vars

class MultiNetworkMutationsFuncs:
    def __init__(self):
        self.sys_vars = sys_vars()  
    
    '''
    update each genome in genome collection with the mutated genome
    NOTE: we have not yet kept track of innovation numbers!
    '''
    def return_mutated_collection(self,genome_collection):
        #initialise a dictionary to keep track of mutatated genes this generation!
        mutated_genes_dict = {}
        #each element of the dict is [source,target,innovation_number]
        #initialise the first mutated gene key (this is a dummy key to save checks later on that a mutation has been added!)
        mutated_genes_dict[0]={'edge_pair':None,'innovation_number':None}
        #add genes we know will be brought up from the random seed
        #loop through each genome descriptor in each species, and update the genome with its mutated version

        '''
        first find the max innovation number initially of all the genomes
        '''
        global_max_innov_number=1
        for genome_descriptors in genome_collection.genomes_dict.values():
            for genome_descriptor in genome_descriptors:
                if len(genome_descriptors)!=0:
                    g = genome_descriptor['genome']
                    #find the maximumum innovation number of that genome (not mutated yet)
                    max_innov_of_g = np.max(g.connection_innov_numbers)
                    if max_innov_of_g>global_max_innov_number:
                        global_max_innov_number=max_innov_of_g


        '''
        then loop through all genomes and mutate each in turn. keeping track of BOTH
        -the global maximum innovation number
        -and keep note of the mutations that have happened
        '''
        for genome_descriptors in genome_collection.genomes_dict.values():
            #loop through each genome descriptor in genome descriptors
            #print('new species')
            for genome_descriptor in genome_descriptors:
                #print(global_max_innov_number)
                #if that species exists
                if len(genome_descriptors)!=0:
                    #extract the genome
                    g = genome_descriptor['genome']
                    #if the genome is a champion, copy into the next generation
                    if genome_descriptor['champion']:
                        genome_descriptor['genome']=g
                        pass
                    #otherwise the genome is not a champion, and we mutate it
                    else:
                        #print(g.connection_innov_numbers)
                        #raise SystemExit
                        #mutate the genome. Take in the list of mutated genes, and the global max innovation number over all genomes
                        mutated_g,mutated_genes_dict,global_max_innov_number = mutation_functions().mutate(g,mutated_genes_dict,global_max_innov_number)
                        #raise SystemExit
                        #update the genome descriptor
                        genome_descriptor['genome']=mutated_g
                        
                    #find the maximumum innovation number after the mutation has taken place, and evaluate if it has incremented the global max
                    max_innov_of_g = np.max(mutated_g.connection_innov_numbers)
                    if max_innov_of_g>global_max_innov_number:
                        global_max_innov_number=max_innov_of_g

        return genome_collection