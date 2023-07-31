# pyNEAT

## Contents
* Introduction
* Usage
* Changes coming
* Installation
* References

## Introduction
A Python implementation of the NEAT (NeuroEvolution of Augmenting Topologies) Algorithm [1] using NumPy. Fast network operations (forward propagation, mutations, mating) using adjacency matrix representations of neural networks. This is a work in progress, other contributors are welcome!

## Usage 
Currently, we have basic user functionality.

* ### Verification: Evolving XORs
  * Running run_XOR_verification.py will perform the XOR validation as described by Stanley and Miikkulainen [1]. 
* ### Pendulum Balancing
  * Running run_pendulum_balancing.py will evolve a network capable of balancing a (single) pendulum subject to a random perturbation (in angle and angular velocity) from the inverted state. _User note: In the current build, it is possible a network may be output not capable of balancing from larger random perturbations. This is not an explicit bug, just a product of the fitness evaluation and how the code outputs the "best" network. The cause is known, and to be fixed in the next version.._
 
## Changes coming
* User interface
* Major overhaul for optimisation. Currently genomes in each generation are stored in species using a nested dictionary approach. The adjacency matrix representation of neural networks will allow for a fully vectorised implementation, along with multithreading.
* Improved data visualisation and animations.
* Refined fitness evaluation for pendulum balancing.
  
## Installation

References
[1] Stanley KO, Miikkulainen R. Evolving neural networks through augmenting topologies. Evol Comput. 2002 Summer;10(2):99-127. doi: 10.1162/106365602320169811. PMID: 12180173.
