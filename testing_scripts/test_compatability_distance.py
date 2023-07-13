#import numpy
import numpy as np

#import the mating functions
from two_genome_functions.mating import mating
from testing_scripts.test_mating_example import test_mate_example
from two_genome_functions.compatability_distance import compatability_distance

#testing mating().mate()
class test_compatability_distance_example:
    def __init__(self):
        self.parent0,self.parent1,self.child = test_mate_example().test_mate()

    def test(self):
        print('Testing delta with parent0 and itself.')
        delta_parent0 =compatability_distance().evaluate_distance(self.parent0,self.parent0)
        print(delta_parent0)
        print('Testing delta with parent1 and itself.')
        delta_parent1 =compatability_distance().evaluate_distance(self.parent1,self.parent1)
        print(delta_parent1)
        print('Testing parent0 and child')
        delta_parent0_and_child =compatability_distance().evaluate_distance(self.parent0,self.child)
        print(delta_parent0_and_child)
        print('Testing child and parent0')
        delta_child_and_parent0 =compatability_distance().evaluate_distance(self.child,self.parent0)
        print(delta_child_and_parent0)
        print('Testing parent1 and child')
        delta_parent1_and_child =compatability_distance().evaluate_distance(self.parent1,self.child)
        print(delta_parent1_and_child)
        pass