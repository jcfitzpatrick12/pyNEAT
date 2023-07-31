# pyNEAT

## Contents
* Introduction
* Usage
* Changes coming
* Installation
* References

## Introduction
A Python implementation of the NEAT (NeuroEvolution of Augmenting Topologies) Algorithm [1] using NumPy. Fast network operations (forward propagation, mutations, mating) using adjacency matrix representations of neural networks. This is a work in progress, other contributors are welcome.

## Usage 
Currently, we have basic user functionality. Users may customise parameters in _sys_vars.py_

* ### Verification: Evolving XORs
  * Running _run_XOR_verification.py_ will perform the XOR validation as described by Stanley and Miikkulainen [1]. __User note: allow_cycles must be set to False, to disable recurrent connections and cycles in the network.__
* ### Pendulum Balancing
  * Running _run_pendulum_balancing.py_ will evolve a network capable of balancing a (single) pendulum subject to a small random perturbation (in angle) from the inverted state. __User note: In the current build, it is possible a network may be output not capable of balancing from larger random perturbations. This is not an explicit bug, just a product of the fitness evaluation and how the code outputs the "best" network. The cause is known, and to be fixed in the next version.__
 
## Changes coming
* Major overhaul for optimisation. Currently genomes in each generation are stored in species using a nested dictionary approach. The adjacency matrix representation of neural networks will allow for a fully vectorised implementation, along with multithreading.
* Creation of a user interface.
* Improved data visualisation and animations.
* Refined fitness evaluation for pendulum balancing.
* Seperation of different parameter variables in _sys_vars.py_ into different scripts depending on the application. This is required, since some values are dependent on whether we are running XOR validation or pendulum balancing, such as allow_cycles or the length of the pendulum.
* Automate pendulum testing of the analytical solutions and, related to the above point, create a seperate sys_vars since the derived solution is dependent on pendulum length for example.
  
## Installation

References
[1] Stanley KO, Miikkulainen R. Evolving neural networks through augmenting topologies. Evol Comput. 2002 Summer;10(2):99-127. doi: 10.1162/106365602320169811. PMID: 12180173.