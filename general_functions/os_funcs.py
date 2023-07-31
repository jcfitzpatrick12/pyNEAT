'''
class which handles os functions such as creating folders, returning folder paths etc.
'''

import os
import inspect
import numpy as np

class osFuncs:
    def get_data_path(self):
        # Get the base folder of pyNEAT using the current file's path
        current_file_path = os.path.abspath(__file__)
        pyNEAT_base_folder = os.path.dirname(os.path.dirname(current_file_path))

        # Construct the path to the "data" folder within the base folder
        data_folder_path = os.path.join(pyNEAT_base_folder, "data")

        # Check if the "data" folder already exists, and create it if not
        if not os.path.exists(data_folder_path):
            os.makedirs(data_folder_path)
            #print("Successfully created 'data' folder in the base folder of pyNEAT.")

        # Output the absolute path of the "data" folder
        #print("Absolute path of the 'data' folder:", data_folder_path)

        return data_folder_path

    
    #numpy save an object, with allow pickle as an option
    def save_data(self,obj_name,obj,**kwargs):
        default_allow_pickle=False
        allow_pickle = kwargs.get('allow_pickle',default_allow_pickle)
        datapath = osFuncs().get_data_path()
        fpath = os.path.join(datapath,obj_name)
        np.save(fpath,obj,allow_pickle=allow_pickle)
        pass

    #numpy load an object, with allow pickle as an option
    def load_data(self,obj_name,**kwargs):
        default_allow_pickle=False
        allow_pickle = kwargs.get('allow_pickle',default_allow_pickle)
        datapath = osFuncs().get_data_path()
        fpath = os.path.join(datapath,f'{obj_name}.npy')

        if allow_pickle:
            obj = np.load(fpath,allow_pickle=allow_pickle).item()
        if not allow_pickle:
            obj=np.load(fpath,allow_pickle=allow_pickle)
        return obj