from .internal.data_procurer import data_procurer as dp
from torch.utils.data import TensorDataset
import torch

class test_data(dp):

    def _load(self, **kwargs) -> str:
        return "RawDataString1Test"
    
    def _process(self, loaded_raw_data, **kwargs) -> list[tuple[TensorDataset]]:
        #bool_tensor_1 = (torch.FloatTensor(10, 4).uniform_() > 0.8)
        #bool_tensor_2 = (torch.FloatTensor(5, 4).uniform_() > 0.2)

        float_tensor_1 = torch.rand(10, 4)
        float_tensor_2 = torch.rand(3, 4)

        training_data = TensorDataset(torch.rand(10, 4, 2), torch.rand(10, 4, 2), float_tensor_1) 
        validation_data = TensorDataset(torch.rand(3, 4, 2), torch.rand(3, 4, 2), float_tensor_2) 
        return [training_data, validation_data]
