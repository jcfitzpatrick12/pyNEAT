
'''
Similar to NEAT testing but opt for a vector implementation

EDGE CASES TO REMEMBER
for add link (with/without cycles) make sure there are no cycles to begin with if applicable
I have written function to determine whether there are cycles in any graph encoded by an adjacency matrix, add a short check to make sure the input graph has no cycles.

To add
-Recurrent connections do not have weights properly visualised, change this
-Add in functions which allow the random activation of previously disabled nodes! How is this done in the formal NEAT implementation?
-When rewriting all the code keep in mind so-called "mutability bugs" (use np.copy(arb_genome.connection_genes) for example).
-When mutating over a population, keep track of the connection genes added. 
when each mutation is performed, add it to a "list" (indeterminate form) to keep track of 
added mutations. If another mutation happens in the same generation, then assign it the same innovation number
as the first. What if the same mutation happens in a DIFFERENT generation? 
Looks like it is just assigned a different innovation number...

Now let's try to write the XOR evolution!
TRY KEEP THINGS GENERAL WITH IN MIND TO APPLY TO PENDULUM BALANCING.
-generate a spread of N networks, and group into species. This will be generation 0
-feedforward each of the networks as appropriate for the operation
-evaluate the shared fitness functions amongst the species.
-perform the mutating, mating operations ect. and find generation i+1
-repeat.

'''

import numpy as np
from test_genome_example import test_genome_example
from test_feedforward_example import test_feedforward_example
from test_mutation_example import test_mutation_example
from test_mating_example import test_mate_example
from test_compatability_distance import test_compatability_distance_example
from feedforward import feedforward
from visualise_genome import visualise_genome


'''
testing genome functions
'''
#test_genome_example().test()
'''
testing the feedforward
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
test mating between Parent1 and Parent2 in the NEAT paper
'''
#parent0,parent1,child = test_mate_example().test_mate()

'''
testing the compatability_distance
'''

test_compatability_distance_example().test()

