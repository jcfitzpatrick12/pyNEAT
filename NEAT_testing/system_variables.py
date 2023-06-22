'''
class which neatly stores the mutation variables
'''

class system_variables:
    def __init__(self):
        '''
        mutation variables
        '''
        self.weight_range=[0,1]
        self.probability_weight_perturbed = 0.8
        self.probability_uniform_weight_perturbation = 0.9
        self.uniform_perturbation_radius = 0.2
        self.probability_new_weight_value = 1-self.probability_uniform_weight_perturbation
        self.probability_add_node = 0.03
        self.small_network_size = 20
        self.probability_add_link_small_network = 0.05
        self.probability_add_link_large_network = 0.3