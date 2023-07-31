'''
Function that takes in a historical record, and plots that parameter over the generations
'''

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors

from general_functions.os_funcs import osFuncs
from sys_vars import sys_vars

import numpy as np
from numpy.random import normal

from plotting.plot_genome import visualise_genome
from one_pendulum_model.balance_pendulum import BalancePendulum
from one_genome_functions.feedforward import feedforward
from plotting.plot_one_pendulum import one_pendulum_plot


class plotNEAT:
    def __init__(self):
        self.fsize=10
        self.fsize_head = 10
        self.fsize_ticks = 10


    def test_XOR_verification(self):
        #load the historical records and plot
        historical_records_dict=osFuncs().load_data('historical_records_dict',allow_pickle=True)
        #plot the history
        self.plot_record_all_species(historical_records_dict,record_type='Maximum Fitness')
        self.plot_record_all_species(historical_records_dict,record_type='Number of Networks')

        #load the most recent proposed_best_genome
        proposed_best_genome = osFuncs().load_data('proposed_best_genome',allow_pickle=True)
        visualise_genome().plot_network(proposed_best_genome)

        output_vector_00 = feedforward().timestep_propagation(proposed_best_genome,[1,0,0],10e-6,'exponential')
        output_vector_01 = feedforward().timestep_propagation(proposed_best_genome,[1,0,1],10e-6,'exponential')
        output_vector_10 = feedforward().timestep_propagation(proposed_best_genome,[1,1,0],10e-6,'exponential')
        output_vector_11 = feedforward().timestep_propagation(proposed_best_genome,[1,1,1],10e-6,'exponential')

        print(f'Network input: {[1,0,0]}; network output: {output_vector_00}; expected_output: {0}')
        print(f'Network input: {[1,0,1]}; network output: {output_vector_01}; expected_output: {1}')
        print(f'Network input: {[1,1,0]}; network output: {output_vector_10}; expected_output: {1}')
        print(f'Network input: {[1,1,1]}; network output: {output_vector_11}; expected_output: {0}')
        pass

    #takes the most recent saved genome and tries to balance a pendulum with it
    def plot_pendulum_run(self):
        #load the historical records and plot
        historical_records_dict=osFuncs().load_data('historical_records_dict',allow_pickle=True)
        #plot the history
        self.plot_record_all_species(historical_records_dict,record_type='Maximum Fitness')
        self.plot_record_all_species(historical_records_dict,record_type='Number of Networks')

        #load the most recent proposed_best_genome
        proposed_best_genome = osFuncs().load_data('proposed_best_genome',allow_pickle=True)
        #visualise this genome
        visualise_genome().plot_network(proposed_best_genome)
        #start the pendulum at some small angle with some small velocity (radians and radians/s)
        y0 = [np.pi-normal(0,sys_vars().initial_theta_normal_width),normal(0,sys_vars().initial_theta_dot_normal_width)]
        #try balance the pendulum with the best proposed genome
        BalancePendulum().try_balance(proposed_best_genome,y0) 

        ### Now load the most recent balance pendulum data
        sol = osFuncs().load_data('sol')
        t = osFuncs().load_data('t')
        base_pos_x = osFuncs().load_data('base_pos_x')
        force_constants = osFuncs().load_data('force_constants')

        pend_plots=one_pendulum_plot(sol,t,base_pos_x,force_constants=force_constants)
        pend_plots.plot_sol()
        pend_plots.animate_pendulum()
        pend_plots.plot_force_constant()

        pass

    #before doing any plotting, we need to pad all the species that were not there at day one!
    def pad_historical_records(self):
        for record_type,records in self.historical_records_dict.items():
            #print(record_type)
            #the number of generations is always assumed to be the number of records for the first generation (otherwise some tracking went wrong)
            max_generation = len(records[0])
            for species_index,record_data in records.items():
                #intialise an array of zeros (which will be the default value for now)
                new_record_data = np.zeros(max_generation)*np.nan
                #find the current length of record_data
                pre_padded_length = len(record_data)
                #place in the non-zero records FROM THE END
                new_record_data[-pre_padded_length::]=record_data
                #update the record
                self.historical_records_dict[record_type][species_index]=new_record_data         
        pass
    

    def plot_record_all_species(self,historical_records_dict,**kwargs):
        self.historical_records_dict = historical_records_dict
        # Pad the historical records
        self.pad_historical_records()


        record_type = kwargs.get('record_type')
        record_dict = self.historical_records_dict[record_type]

        #print(record_dict)

        # Custom colormap: first color is red, the rest is from viridis
        colors = [(1, 0, 0)] + [plt.cm.viridis(i) for i in range(256)]
        new_colormap = mcolors.LinearSegmentedColormap.from_list("red_viridis", colors, N=257)

        # Preparing the data for the heatmap
        species = list(record_dict.keys())
        generations = list(range(1, len(next(iter(record_dict.values()))) + 1))
        
        heatmap_data = np.array([record_dict[spc] for spc in species])
        
        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.imshow(heatmap_data, cmap=new_colormap, aspect="auto")

        # Setting the ticks
        ax.set_xticks(np.arange(len(generations)))
        ax.set_yticks(np.arange(len(species)))

        # Labeling the ticks
        ax.set_xticklabels(generations)
        ax.set_yticklabels(species)

        # Rotating the tick labels for better visibility
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Adding a colorbar
        cbar = fig.colorbar(cax)
        cbar.set_label(record_type)

        ax.set_xlabel('Generation')
        ax.set_ylabel('Species Index')

        plt.tight_layout()
        plt.show()

        pass
        
