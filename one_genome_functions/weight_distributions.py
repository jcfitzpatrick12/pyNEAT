from numpy.random import uniform,normal
from sys_vars import sys_vars
class WeightDistributions:
    def __init__(self,):
        self.sys_vars = sys_vars()

    def return_weights(self,requested_size):
        if self.sys_vars.requested_weight_distribution=="normal":
            return normal(self.sys_vars.normal_loc_width[0],self.sys_vars.normal_loc_width[1],size=requested_size)
        if self.sys_vars.requested_weight_distribution=="uniform":
            return uniform(self.sys_vars.uniform_weight_range[0],self.sys_vars.uniform_weight_range[1],size=requested_size)