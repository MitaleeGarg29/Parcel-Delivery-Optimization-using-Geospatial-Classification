import torch
import numpy as np
import pandas as pd
import requests
import json

from torch.utils.data import TensorDataset

from .internal.data_procurer import data_procurer as dp

class template(dp):
    """
    An empty data procurer subclass serving as an example on how to define a custom data procurer. May also serve as a boilerplate.
    """

    def _load(self, **kwargs) -> str:
        text_data = ""

        # TODO: load data in string format

        return text_data

    
    def _process(self, loaded_raw_data, **kwargs) -> list[tuple[TensorDataset]]:

        processed_training_data_input_1_shifted = None
        processed_training_data_input_2 = None
        processed_training_data_target = None

        processed_validation_data_input_1_shifted = None
        processed_validation_data_input_2 = None
        processed_validation_data_target = None

        # TODO: Process loaded_raw_data string into final state

        training_data = TensorDataset(processed_training_data_input_1_shifted, processed_training_data_input_2, processed_training_data_target) 
        validation_data = TensorDataset(processed_validation_data_input_1_shifted, processed_validation_data_input_2, processed_validation_data_target) 
        
        return [training_data, validation_data]
    
    # optional; only used for CSV-creation
    def _set_data_headers(self):
        data_headers = [
            ["Input 1, Column 1", "Input 1, Column 2", "Input 1, Column 3", "Input 1, Column 4", "..."],
            ["Input 2, Column 1", "Input 2, Column 2", "Input 2, Column 3", "Input 2, Column 4", "..."],
            #...
            ["Label, Column 1", "Label, Column 2", "..."]
        ]

        return data_headers

