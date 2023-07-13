import numpy as np

'''
class which 
-stores the dictionary of available activation functions
-returns the requested activation function.
'''
class activation_functions:
    #given an input of node(s) a, a of arbitrary shape, return the output of that (those) node(s)
    #linear activation function (returns the input unchanged)
    def linear_activatation_function(self,a):
        return a
    #expononitial activation function
    def exponential_activation_function(self,a):
        return 1/(1+np.exp(-4.9*a))
    
    #takes an input, and the requested activation function and returns the output of that neuron
    def return_requested_activation_function(self,requested_activation_function):
        activation_functions_dict = {'linear' : self.linear_activatation_function, 'exponential' : self.exponential_activation_function}
        func = activation_functions_dict[requested_activation_function]
        return func