
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

To note:

It looks like in NEAT paper they use 1-indexing for innovation numbers where I have assumed 0-indexing everywhere
My 0-indexing and method of padding with nans has made writing the mating function horrible
May have to rework the "padding" and not use nans.
Instead, best way to do this may be to infact keep track of how many enabled nodes simply through THE ENABLE BITS IN NODE GENES
And instead of 0-indexing, use 1-indexing for the innovation numbers.
This will mean I don't have to do any jiggery pokery with the mating functions.
Set all unused node descriptors to be ZERO and we can tell which nodes are enabled through the node enable bits.

'''

import numpy as np
from test_genome_example import test_genome_example
from test_feedforward_example import test_feedforward_example
from test_mutation_example import test_mutation_example
from test_mating_example import test_mate_example
from feedforward import feedforward
from visualise_genome import visualise_genome


'''
mutation unit testing for a single network
'''

'''
test_genome_example().test()
test_feedforward_example().test()
test_feedforward_example().test_recurrent()
mutated_genome = test_mutation_example().test_single_mutation()
#test the mutated_genome for a feedforward
output_vector= feedforward().timestep_propagation(mutated_genome,[0.3,0.5,0.8],10e-6,'linear')
print(output_vector)
'''

'''
test mating between Parent1 and Parent2 in the NEAT paper
'''
child = test_mate_example().test_mate()
visualise_genome().plot_network(child)

