
'''
Similar to NEAT testing but opt for a vector implementation

-implement input verifications 

EDGE CASES TO REMEMBER
for add link (with/without cycles) make sure there are no cycles to begin with if applicable

To add
-Recurrent connections do not have weights properly visualised, change this
-Add in functions which allow the random activation of previously disabled nodes! How is this done in the formal NEAT implementation?
-When rewriting all the code keep in mind so-called "mutability bugs" (use np.copy(arb_genome.connection_genes) for example).
'''

import numpy as np
from test_feedforward_example import test_feedforward_example
from test_mutation_example import test_mutation_example
from test_genome_example import test_genome_example
from genome import genome_builder
from genome_functions import genome_functions
from visualise_genome import visualise_genome


'''
mutation unit testing for a single network
'''
#test_genome_example().test()
#test_feedforward_example().test()
#test_feedforward_example().test_recurrent()
mutated_genome = test_mutation_example().test_single_mutation()
#test the mutated_genome for a feedforward
#output_vector= genome_functions(mutated_genome).feedforward([1,1,1],10e-6,'linear')
#print(output_vector)
