import json
import argparse
import torch
import numpy as np

from torch.utils.data import TensorDataset

from .internal.data_procurer import data_procurer as dp
from .internal.data_processing import data_processing_util


class test_data_V2(dp):

    def _load(self, **kwargs) -> str:
        return "Test"

    
    def _process(self, loaded_raw_data, **kwargs) -> list[tuple[TensorDataset]]:
        
        output = []

        user_count = 3
        for index in range(1, user_count + 1):
            training_data = TensorDataset(
                torch.tensor(range(0, index*10, index)),
                torch.tensor(range(0, index*10, index)),
                torch.tensor(range(0, index*10, index))
            )

            validation_data = TensorDataset(
                torch.tensor(range(0, index*5, index)),
                torch.tensor(range(0, index*5, index)),
                torch.tensor(range(0, index*5, index))
            )

            output.append([training_data, validation_data])

        return output
    
    
    def _set_data_headers(self):
        data_headers = [
            #["Lat", "Long", "From Home", "From POI 1", "From POI 2", "From POI 3", "From POI 4", "From POI 5", "From POI 6", "From POI 7", "From POI 8", "FromPOI 9", "From POI 10"],
            ["TS Input Day"],
            ["Future Covariant Day"],
            ["Label day"]
        ]

        return data_headers
