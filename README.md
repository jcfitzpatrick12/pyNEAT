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
  * Running _run_pendulum_balancing.py_ will evolve a network capable of balancing a (single) pendulum subject to a small random perturbation (in angle) from the inverted state. __User note: In the current build, it is possible a network may be output not capable of balancing from larger random perturbations. This is not an explicit bug, just a product of the fitness evaluation and how the code outputs the "best" network. The cause is known, and to be fixed in the next version. Note that setting allow_cycles to False may make it more difficult to find a solution.__
 
## Changes coming
* Major optimisation overhaul. Currently genomes in each generation are stored in species using a nested dictionary approach. The adjacency matrix representation of neural networks will allow for a fully vectorised implementation, along with multithreading.
* Creation of a user interface.
* Improved data visualisation and animations.
* Refined fitness evaluation for pendulum balancing.
* Seperation of _sys_vars.py_ into different scripts depending on the application. This is required, since some variables currently need manually altered whether we are running XOR validation or pendulum balancing, such as allow_cycles.
* Automate pendulum testing of the analytical solutions and, related to the above point, create a seperate sys_vars since the derived solution is dependent on pendulum length for example.
  
## Installation
We describe installation for Windows, and will soon to expand to other operating systems. You must ensure that conda is installed on your system. You can verify this by typing conda --version in your command prompt. If you've recently installed conda, you may need to close and reopen the command prompt to recognize the conda command or ensure that conda is added to your PATH. Replace _path_to_directory_ with the path on your system where you'd like to clone the repository.

_Navigate to your desired directory in the command prompt:_ \
cd path_to_directory \
_Clone the pyNEAT repository:_ \
git clone https://github.com/jcfitzpatrick12/pyNEAT.git \
_Navigate to the cloned directory where environment.yml is stored:_ \
cd pyNEAT \
_Create and set up the conda environment using the provided environment.yml file:_ \
conda env create -f environment.yml \
_To verify the environment has been installed, activate the environment:_ \
conda activate pyNEAT-env 

With the enviroment activated, you can refer to __Usage__ section to get started! When executing the code, ensure that the active Python interpreter corresponds to the conda environment that was created. This ensures dependencies and libraries are appropriately resolved from the correct environment.

## References
[1] Stanley KO, Miikkulainen R. Evolving neural networks through augmenting topologies. Evol Comput. 2002 Summer;10(2):99-127. doi: 10.1162/106365602320169811. PMID: 12180173.
