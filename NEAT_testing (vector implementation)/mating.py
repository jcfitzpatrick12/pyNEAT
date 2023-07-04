'''
class which handles mating between two genomes
'''

import numpy as np
#directly draw out the uniform distribution sampling
from numpy.random import uniform,randint
from sys_vars import sys_vars
from genome import genome
from two_genome_functions import two_genome_functions

#takes in two genome_functions
class mating(two_genome_functions):
    def __init__(self):
        self.sys_vars=sys_vars()

    #define the mating functions
    '''
    takes in two parents and their respective fitness, and outputs child using innovation number matching as in the NEAT paper.

    See the below NEAT paper extracts:

    "[using innovation numbers] the system now
    knows exactly which genes match up with which (Figure 4). When crossing over, the
    genes in both genomes with the same innovation numbers are lined up. These genes
    are called matching genes. Genes that do not match are either disjoint or excess, depend-
    ing on whether they occur within or outside the range of the other parent’s innovation
    numbers."

    "In composing the offspring, genes are randomly chosen from either parent at matching genes,
    whereas all excess or disjoint genes are always included from the more fit parent."

    "There was a 75% chance that an inherited gene was disabled if it was disabled
    in either parent."
    '''

    def mate(self,parent0,parent1,fitness_parent0,fitness_parent1):
        #if fitter parent = np.nan, this means that both are equal fitness networks
        fitter_parent_index=np.nan
        #if one is fitter than the other, note this
        if fitness_parent0>fitness_parent1:
            fitter_parent_index=0
        elif fitness_parent1>fitness_parent0:
            fitter_parent_index=1

        #create the combined connection-genes array.
        combined_connection_genes = two_genome_functions().combine_genome_connection_genes(parent0,parent1)
        #find the disjoint and excess genes
        #these occur if innovation number in one parent != the innovation number of another at the same edge element in the adjacency matrix
        #first we find the genes which contain either a disjoint or excess gene IN EITHER PARENT.
        #we will not yet verify which are disjoint and which are excess
        disjoint_or_excess_genes = two_genome_functions().find_disjoint_or_excess_genes(combined_connection_genes)
        #extract the number of dijoint or excess nodes in each parent
        #num_disjoint_parent0,num_excess_parent0 = self.num_disjoint_and_excess(0,disjoint_or_excess_genes[...,0],combined_connection_genes)
        #num_disjoint_parent1,num_excess_parent1 = self.num_disjoint_and_excess(1,disjoint_or_excess_genes[...,1],combined_connection_genes)
        #build the childs connection_genes
        child_connection_genes = self.build_child_connection_genes(combined_connection_genes,disjoint_or_excess_genes,fitter_parent_index)
        #build the childs node_genes
        child_node_genes = self.build_child_node_genes(child_connection_genes,parent0,parent1)
        #birth the child!
        child=genome(child_node_genes,child_connection_genes)
        #slice the child to be the minimum size
        sliced_child = child.return_enabled_genome()
        #okay so we now have the indices of matching genes and of the disjoint/excess genes
        #we can now create the (minimum size) sliced
        return sliced_child
    
    '''
    implicit in the building of the childrens node genes is that to mate two parents, the networks must have
    identical numbers of inputs and outputs.
    '''
    def build_child_node_genes(self,child_connection_genes,parent0,parent1):
        #pad out the smaller network so we can better match up the two directly encoded networks
        #first find the parent with the largest number of nodes.
        num_global_nodes_parent0 = parent0.num_nodes(type='global_max')
        num_global_nodes_parent1 = parent1.num_nodes(type='global_max')
        #find the parent with the max number of num_global_max_nodes
        max_num_global=max(num_global_nodes_parent0,num_global_nodes_parent1)
        #find which parent this belongs to and first simply assign the childrens node genes to the largest possible case
        if num_global_nodes_parent0>=num_global_nodes_parent1:
            upper_bound_child_node_genes = parent0.node_genes
        else:
            upper_bound_child_node_genes=child_node_genes = parent1.node_genes

        #we can now shrink child_node_genes depending on how many nodes are actually used in child_connection_genes
        #first sum over the rows of child_connection_genes innovation numbers:
        column_sum = np.sum(child_connection_genes[...,2],axis=0)
        #and then sum over the rows
        row_sum = np.sum(np.vstack(child_connection_genes[...,2]),axis=-1)
        #evaluate the sum of columns and rows. If this is non-zero, this tells us that node is being used at least somewhere
        node_sum = column_sum+row_sum
        #reverse the array
        node_sum=node_sum[::-1]
        #find the index of the first non zero element, and denote this N. Then max_num_global-N will give us the minimum number of 
        #necessary nodes to describe the childs genes
        N=(node_sum!=0).argmax(axis=0)
        #thus, we find the minimum number of childrens nodes we need to describe the new network
        num_childrens_nodes = max_num_global-N
        #and we can safely slice childrens_node_genes at this value
        child_node_genes=upper_bound_child_node_genes[:num_childrens_nodes,:]
        return child_node_genes
    

    #input the combined_connection_genes, the combined_location of disjoint or excess genes (parents indexed over the last axis of both arrays) and the index of the fitter parent
    #whether that is parent 1 or 0
    def build_child_connection_genes(self,combined_connection_genes,disjoint_or_excess_genes,fitter_parent_index):
        #first presume that both parents are equal fitness (which may not be true) and randomly select ALL genes from either parents
        #create a random array of ones and zeros (in the shape of (num_global_nodes,num_global_nodes))
        random_array_of_ones_and_zeros = randint(0,2,size=np.shape(disjoint_or_excess_genes[...,0]))
        #array that holds "true" if that connection gene is taken from parent one
        true_if_we_take_genes_from_parent_one = np.array(random_array_of_ones_and_zeros,dtype=bool)
        #array that holds "true" if that connection gene is taken from parent zero
        true_if_we_take_genes_from_parent_zero = ~np.array(random_array_of_ones_and_zeros,dtype=bool)
        #copy these elements into a vectorised array
        true_if_we_take_genes_from_parent_one_vect = np.repeat(true_if_we_take_genes_from_parent_one[...,np.newaxis],3,axis=2)
        true_if_we_take_genes_from_parent_zero_vect = np.repeat(true_if_we_take_genes_from_parent_zero[...,np.newaxis],3,axis=2)
        #add all genes at random from either parent!
        child_connection_genes = np.multiply(true_if_we_take_genes_from_parent_zero_vect,combined_connection_genes[...,0])+np.multiply(true_if_we_take_genes_from_parent_one_vect,combined_connection_genes[...,1])
        '''
        now deal if one parent is fitter than the other (force reassign the disjoint or excess genes to be from the fitter parent)
        '''
        #if we do not have equal fitness parents
        #reassign the disjoint or excess genes to be from the fitter parent
        if fitter_parent_index in [0,1]:
            #print(fitter_parent_index)
            #reassign all the disjoint or excess genes to be acquired from the fitter parent!
            disjoint_or_excess_in_either_parent=disjoint_or_excess_genes[...,0]+disjoint_or_excess_genes[...,1]
            child_connection_genes[disjoint_or_excess_in_either_parent]=combined_connection_genes[disjoint_or_excess_in_either_parent,:,fitter_parent_index]
            pass
        
        '''
        "There was a 75% chance that an inherited gene was disabled if it was disabled
        in either parent."
        -it would seem my method of using enable bits is not fantastic
        -as it cannot distinguish between those genes which are "auto" disabled (pre-assigned zero)
        -and those that are MANUALLY assigned to be disabled
        -easy workaround using extra information that an "intentionally" disabled gene will have non zero innovation number
        '''

        #find two arrays, where, for each parent, that gene has one if the enable bit is INTENTIONALLY zero.
        #output of the below four lines is an array that holds true if the gene was INTENTIONALLY disabled by either parent (i.e not preassigned to be zero!)
        true_if_enable_bit_is_zero = combined_connection_genes[...,1,:]==0
        true_if_innov_number_is_non_zero = combined_connection_genes[...,2,:]>0
        true_if_enable_bit_is_intentionally_zero = np.multiply(true_if_enable_bit_is_zero,true_if_innov_number_is_non_zero)
        true_if_enable_bit_is_intentionally_zero_in_either_parent = (true_if_enable_bit_is_intentionally_zero[...,0]+true_if_enable_bit_is_intentionally_zero[...,1])>0

        #now for each gene that holds "true" reassign the enable bit to be zero if the rv at that gene is <25pc
        rv = uniform(0,1,size=np.shape(true_if_enable_bit_is_intentionally_zero_in_either_parent))
        true_if_we_enable_that_gene = rv<self.sys_vars.probability_enable_gene
        true_if_we_enable_that_gene_and_it_is_a_gene_we_can_change = np.multiply(true_if_we_enable_that_gene,true_if_enable_bit_is_intentionally_zero_in_either_parent)
        #enable those genes!
        child_connection_genes[true_if_we_enable_that_gene_and_it_is_a_gene_we_can_change,1]=1
        #return the childrens connection genes
        return child_connection_genes
    
    