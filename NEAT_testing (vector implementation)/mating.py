'''
class which handles mating between two genomes
'''

import numpy as np
#directly draw out the uniform distribution sampling
from numpy.random import uniform,randint
from sys_vars import sys_vars

class mating:
    def __init__(self,):
        self.sys_vars=sys_vars

    #function which returns the combined parents connection genes (padding where necessary so both parents genes are the same shape, even if the networks have different topologies)
    #of dimension (num_global_max_nodes_both_parents,num_global_max_nodes_both_parents,3,2)
    #where the zeroth element last axis is parent0 and the first element last axis is parent1
    def combine_parents_connection_genes(self,parent0,parent1):
        #pad out the smaller network so we can better match up the two directly encoded networks
        #first find the parent with the largest number of nodes.
        num_global_nodes_parent0 = parent0.num_nodes(type='global_max')
        num_global_nodes_parent1 = parent1.num_nodes(type='global_max')
        #if parent 0 is larger , pad parent 1
        if num_global_nodes_parent0>num_global_nodes_parent1:
            parent1=parent1.return_padded_genome(num_nodes=num_global_nodes_parent0)
        #if parent 1 is larger , pad parent 0
        elif num_global_nodes_parent1>num_global_nodes_parent0:
            parent0=parent0.return_padded_genome(num_nodes=num_global_nodes_parent1)

        #now both parents have arrays of the same shapes!
        #make a combined matrix of the connection_genes (so we can efficiently extract which gene to take from which using indices)
        #place parent0 into the 0 element of the last axis
        #place parent1 into the 1 element of the last axis
        combined_connection_genes = np.zeros(np.shape(parent0.connection_genes)+(2,))
        combined_connection_genes[...,0]=parent0.connection_genes
        combined_connection_genes[...,1]=parent1.connection_genes
        return combined_connection_genes
    
    def find_matching_genes(self,combined_connection_genes):
        #since all innovation numbers are integers, if a connection gene shares an innovation number
        #between the parents, their difference should be zero!
        #otherwise, all other connections are assigned to be nan valued
        diff_innovation_numbers = combined_connection_genes[...,-1,0]-combined_connection_genes[...,-1,1]
        #matching genes will hold one if that connection gene is SHARED by both parents
        matching_genes = (~np.array(diff_innovation_numbers,dtype=bool))
        return matching_genes
    
    #given the combined connection_genes, identify disjoint or excess genes.
    def find_disjoint_or_excess_genes(self,combined_connection_genes):
        #first determine which elements are nans. Holds 1 if that connection gene has nan innovation number
        innov_number_is_nan_boole = np.isnan(combined_connection_genes[...,-1,:])*1
        #sum over the last axis of innov_number_is_nan
        #if that connection_gene is nan in one parent, and non-nan in the other parent then
        #the sum will be one! 
        #if that connection_gene is nan in both parents, then the sum will be 2
        #if that connection gene is an integer in both parents then the sum will be zero.
        innov_number_is_nan_boole_summed = np.sum(innov_number_is_nan_boole,axis=-1)
        #now evaluate which connection_genes are disjoint or excess genes (since we have summed over the last axis)
        #we have lost information as to WHICH parent these belong too, this will not be difficult to recover this information
        #holds true if that connection gene is either disjoint or excess in EITHER ONE of the parents
        disjoint_or_excess_genes = (innov_number_is_nan_boole_summed==1)
        #find out whether the disjoint or excess genes are specifically
        disjoint_or_excess_parent0 = np.multiply(disjoint_or_excess_genes,combined_connection_genes[...,2,0])>0
        disjoint_or_excess_parent1 = np.multiply(disjoint_or_excess_genes,combined_connection_genes[...,2,1])>0
        return disjoint_or_excess_parent0,disjoint_or_excess_parent1
    
    #takes in the parents index, where the disjoint or excess genes are for that parent, and the combined connection_genes
    def num_disjoint_and_excess(self,requested_parent_index,disjoint_or_excess_genes,combined_connection_genes):
        #find the other parents index
        other_parent_index = 1-requested_parent_index
        max_innov_other_parent= np.nanmax(combined_connection_genes[...,2,other_parent_index])
        #extract the connection genes of the requested parents index
        requested_parent_connection_genes = combined_connection_genes[...,requested_parent_index]
        #extract specifically disjoint or excess genes (only the innovation numbers)
        disjoint_or_excess_gene_innovation_numbers = requested_parent_connection_genes[disjoint_or_excess_genes,2]
        #an array which holds ones for only disjoint genes
        disjoint_genes = disjoint_or_excess_gene_innovation_numbers<=max_innov_other_parent
        excess_genes = disjoint_or_excess_gene_innovation_numbers>max_innov_other_parent
        #return the number of disjoint and excess genes respectively
        return np.nansum(disjoint_genes),np.nansum(excess_genes)
        #return num_disjoint,num_excess

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
        #if fitter parent = -1, this means that both are equal fitness networks
        fitter_parent_index=np.nan
        #if one is fitter than the other, note this
        if fitness_parent0>fitness_parent1:
            fitter_parent_index=0
        elif fitness_parent1>fitness_parent0:
            fitter_parent_index=1

        #create the combined connection-genes array.
        combined_connection_genes = self.combine_parents_connection_genes(parent0,parent1)

        #find the disjoin and excess genes
        #these occur if a connection gene in one parent is assigned "nan" and the other is assigned an integer
        #first we find the genes which contain either a disjoint or excess gene IN EITHER PARENT.
        #we will not yet verify which are disjoint and which are excess
        disjoint_or_excess_genes_parent0,disjoint_or_excess_genes_parent1 = self.find_disjoint_or_excess_genes(combined_connection_genes)

        '''
        num_disjoint_parent0,num_excess_parent0 = self.num_disjoint_and_excess(0,disjoint_or_excess_genes_parent0,combined_connection_genes)
        num_disjoint_parent1,num_excess_parent1 = self.num_disjoint_and_excess(1,disjoint_or_excess_genes_parent1,combined_connection_genes)
       
        '''

        #birth the child!
        child = self.birth_child(combined_connection_genes,disjoint_or_excess_genes_parent0,disjoint_or_excess_genes_parent1,fitter_parent_index)
        #okay so we now have the indices of matching genes and of the disjoint/excess genes
        #we can now create the child genome
        return child
    
   #takes in the combined connection genes, the location of the matching_genes and disjoint/excess genes, and the index of the fitter parent
    #and generates the childs genome

    '''
    "Matching genes are inherited
    randomly, whereas disjoint genes (those that do not match in the middle) and excess
    genes (those that do not match in the end) are inherited from the more fit parent. In
    this case, equal fitnesses are assumed, so the disjoint and excess genes are also inherited
    randomly. The disabled genes may become enabled again in future generations: there’s
    a preset chance that an inherited gene is disabled if it is disabled in either parent." -NEAT paper
    '''

    def birth_child(self,combined_connection_genes,disjoint_or_excess_genes_parent0,disjoint_or_excess_genes_parent1,fitter_parent_index):
        np.random.seed(1)

        print(combined_connection_genes[...,0,0])
        print(combined_connection_genes[...,1,0])
        print(combined_connection_genes[...,2,0])
        print()
        print(combined_connection_genes[...,0,1])
        print(combined_connection_genes[...,1,1])
        print(combined_connection_genes[...,2,1])

        raise SystemExit
        #we need to take special care with where the innovation number is zero! since later, we set particular innovations numbers of value nan to zero
        #we need some way to remember which was the original zero! make a note of this now...
        where_original_innov_zero = combined_connection_genes[...,2,0]==0
        #create a random array of ones and zeros (in the shape of (num_global_nodes,num_global_nodes)
        random_array_of_ones_and_zeros = randint(0,2,size=np.shape(disjoint_or_excess_genes_parent0))
        #assuming we are placing ALL genes into the childs connection genome randomly from each parent,
        #we may interpret this matrix as holding the index of the parent we will take that gene from
        #then selectively choose the disjoint or excess genes from the fitter parent if indeed one parent is fitter than the other.
        #array that holds "true" if that connection gene is taken from parent one
        true_if_we_take_genes_from_parent_one = np.array(random_array_of_ones_and_zeros,dtype=bool)
        #array that holds "true" if that connection gene is taken from parent zero
        true_if_we_take_genes_from_parent_zero = ~np.array(random_array_of_ones_and_zeros,dtype=bool)
        #copy these elements into a vectorised array
        true_if_we_take_genes_from_parent_one_vect = np.repeat(true_if_we_take_genes_from_parent_one[...,np.newaxis],3,axis=2)
        true_if_we_take_genes_from_parent_zero_vect = np.repeat(true_if_we_take_genes_from_parent_zero[...,np.newaxis],3,axis=2)
        #now, we have an issue were integer + nan = nan so that for excess and disjoint genes, the innovation number will not be accounted for
        #in the pairwise multiplication and addition since for one parent, the innovation number will be nan.
        #thus, for disjoint or excess genes in parent0, in the corresponding genes in parent1 convert the innovation number to zero
        #identically for parent1. This will mean the disjoint or excess genes will be accounted for.
        combined_connection_genes[disjoint_or_excess_genes_parent0,:,1]=0
        combined_connection_genes[disjoint_or_excess_genes_parent1,:,0]=0

        #add all genes at random from either parent!
        child_connection_genes = np.multiply(true_if_we_take_genes_from_parent_zero_vect,combined_connection_genes[...,0])+np.multiply(true_if_we_take_genes_from_parent_one_vect,combined_connection_genes[...,1])

        #find where the childs innovation numbers are zero, and change them to be nan
        child_has_zero_innov = child_connection_genes[...,2]==0
        child_connection_genes[child_has_zero_innov,2]=np.nan
        #relabel the original zero innovation gene to have zero again!
        child_connection_genes[where_original_innov_zero]=0

        print(child_connection_genes[...,0])
        print(child_connection_genes[...,1])
        print(child_connection_genes[...,2])
        raise SystemExit
        '''
        build the childs node genes
        '''
        return child