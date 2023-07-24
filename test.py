import numpy as np
from testing_scripts.test_genome_example import test_genome_example
from testing_scripts.test_feedforward_example import test_feedforward_example
from testing_scripts.test_mutation_example import test_mutation_example
from testing_scripts.test_mating_example import test_mate_example
from testing_scripts.test_compatability_distance import test_compatability_distance_example
from testing_scripts.test_matching_disjoint_excess_genes import test_matching_disjoint_excess_example
from one_genome_functions.feedforward import feedforward




'''
testing genome functions
'''
#test_genome_example().test()
'''
testing the feedforward functions
'''
#test_feedforward_example().test()
#test_feedforward_example().test_recurrent()
'''
testing mutations for a single network
'''
#mutated_genome = test_mutation_example().test_single_mutation()
#output_vector= feedforward().timestep_propagation(mutated_genome,[0.3,0.5,0.8],10e-6,'linear')
#print(output_vector)

'''
testing the matching genes, disjoint genes and excess genes functions
'''

#test_matching_disjoint_excess_example().test()


'''
test mating between Parent1 and Parent2 in the NEAT paper
'''
#parent0,parent1,child = test_mate_example().test_mate()

'''
testing the compatability_distance
'''
test_compatability_distance_example().test()