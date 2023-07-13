'''
script which will allow us to deal with generations of networks:

-final goal is to hold all the genomes in a single multidim array:
i.e.
node_genes where the max size is the max number of nodes over all the networks
connection_genes where the max size (i.e. max number of rows/cols) is also the max number of nodes over all the networks.
this can be achieved by rewriting all the code except with (say) a multi_genome object.

need to think... because all my functions are geared for so-called "genome" objects.
for now just consider a pythonic array of class objects! Slow, but I can work out the NEAT algorithm then move to optimise it.

DONE
-initialise N networks of identical topologies with random weights

NOT YET DONE
-run feedforward given an (elsewhere defined) function that takes in inputs and performs timestep propagation on each of the networks (output is the network output for each input)
-Evaluate the fitness of each network.
-Function which takes in N genomes in n species, and places them into m new species (based on the paper).
-Evaluate the adjusted fitness of each network
-Function which takes in N genomes placed into m species, and their respective adjusted fitness and generate the offspring 
(offspring are by default in the species they are derived from)
-Function which takes in N genomes placed in the m species, and mutates each in turn. 
'''


        

