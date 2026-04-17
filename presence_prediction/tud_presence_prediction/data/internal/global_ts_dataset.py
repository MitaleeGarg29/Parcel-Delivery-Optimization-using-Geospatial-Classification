from math import ceil
from torch.utils.data import Dataset
import torch

from .data_processing import data_processing_util

class global_timeseries_dataset(Dataset):
    r"""
    Provides a PyTorch-compatible way to generate equally sized samples sequences from multiple PyTorch Datasets of different lengths by applying a custom-sized rolling window view to it. These sequences may then be automatically aquired by a PyTorch DataLoader. The original datasets are not modified directly and are only used to create slices.
    
    The given settings determine how the samples are generated from the original sequences. They are a slice of the size of the window and each slice is offset by the given stride count compared to the previous sample. If a given dataset is too small to generate a single sample, it may optionally get padded.
    
    No attributes within this class should be altered directly externally as that invalidates the idx_lookup required to retrieve the correct sequences. 
    """

    def __init__(self, datasets, window_size, stride, stride_compensation="lower", window_size_compensation="padding", window_size_compensation_limit=-1, padding_dim=0):
        r"""
        Initializes the dataset. 

        Args:
            datasets (list[Dataset]): 
                List of PyTorch Datasets. Each of them must be capable of returning slices of the same unit as window_size and stride. For example, if one unit in the index of each of the Datasets corresponds to a single day, then window_size and stride must also be given in days.
            window_size (int):
                The size a single sample extracted from this dataset should have. When a sample from this dataset is requested, a continous sequence of this length, consisting of multiple samples from one of the internal datasets is returned. Can not be smaller than 1. 
            stride (int):
                The size the step between each sample created by this dataset should have. This determines how far apart the starting positions of two neighboring sample sequences are in the internal dataset. If set to 1, then all possible combinations will be iterated. Higher values result in fewer variations. Setting this equal to window size is effectively equivalent to torch.split(dataset, window_size) and the datasets are split in equal chunks of size window_size. Can not be lower than 1 and should not be higher than window_size.
            stride_compensation (string):
                Controls how this dataset should treat the last sample at the end of an internal dataset if it doesn't entirely cover the window for that window position. For a stride of one, this never occurs.
                Options:
                    - "lower": Lowers the last stride, effectively adjusting the last window position to reach exactly to the end of the dataset.
                    - "drop": Drops the last sequence.
            window_size_compensation (string):
                Controls how this dataset should treat an internal dataset if it doesn't have the minimum length of "window_size" to create one sample sequence.
                Options:
                    - "padding": The existing dataset will be padded to cover exactly one "window_size", and exactly one sample will be created from that dataset.
                    - "drop": The dataset will be dropped.
            window_size_compensation_limit (int):
                Controls how large a dataset must be to be eligable for "padding" window size compensation. If the dataset is smaller than this value it is dropped. Has no effect if padding is not used. 
            padding_dim (int):
                Controls which dimension of the generated sample slice should be padded. Has no effect if padding is not used. 
        """

        if window_size < 1 or type(window_size) != int: raise ValueError("Invalid window size value.")
        if stride < 1 or type(stride) != int: raise ValueError("Invalid stride value.")

        self.datasets = datasets
        self.window_size = window_size
        self.stride = stride
        self.stride_compensation = stride_compensation
        self.window_size_compensation = window_size_compensation
        self.window_size_compensation_limit = window_size_compensation_limit
        self.padding_dim = padding_dim
        self.dataset_variations = []
        self.dataset_paddings = []

        self.update_idx_lookup()

    def get_window_size(self):
        return self.window_size

    def set_window_size(self, new_window_size):
        self.window_size = new_window_size
        self.update_idx_lookup()
    
    def get_stride(self):
        return self.stride
    
    def set_stride(self, new_stride):
        self.stride = new_stride
        self.update_idx_lookup()

    
    def update_idx_lookup(self):
        self.dataset_variations = []
        self.dataset_paddings = []
        self.idx_lookup = []

        for dataset_index, dataset in enumerate(self.datasets):
            i_dataset_idx_infos = []

            i_projected_variations = None
            added_padding = 0

            i_dataset_len = len(dataset) if dataset != None else 0
            # Any None given in the list is treated as an empty dataset
            if dataset == None:
                i_projected_variations = 0
            # if dataset length is lower than window length, no regular sample can be created
            elif i_dataset_len < self.window_size:

                # a single sample can be created from this dataset by adding padding when the tensor is requested                            
                if self.window_size_compensation == "padding": 
                    added_padding = self.window_size - i_dataset_len
                    i_dataset_idx_infos.append(self.generate_sequence_lookup(dataset_index, 0, self.window_size, added_padding))
                    i_projected_variations = 1

                # no sample can be created or no sample should be created because the amount of needed padding would exceed desired limits
                if self.window_size_compensation == "drop" or (i_dataset_len < self.window_size_compensation_limit and self.window_size_compensation=="padding"): 
                    i_projected_variations = 0
            # if dataset length is equal or greather than window length, one or more samples can be created from it. Padding is never used in this case.
            else:
                i_projected_variations = ceil((i_dataset_len - self.window_size) / self.stride) + 1      

                i_current_offset = 0
                dataset_exhaused = False
                while dataset_exhaused == False:
                    i_dataset_idx_infos.append(self.generate_sequence_lookup(dataset_index, i_current_offset, i_current_offset + self.window_size))

                    # check if next sample would still be within the datasets limits    
                    next_stride = self.stride
                    remaining_offset = i_dataset_len - self.window_size - i_current_offset
                    if remaining_offset > 0:
                        # if the full stride can not be used, create last sample with smaller stride
                        if remaining_offset < self.stride:
                            if self.stride_compensation == "lower":
                                next_stride = i_dataset_len - self.window_size - i_current_offset
                            elif self.stride_compensation == "drop":
                                next_stride = 0
                                dataset_exhaused = True
                        if next_stride != None:
                            i_current_offset += next_stride
                    else:
                        dataset_exhaused = True

                # sanity checks
                if i_current_offset != len(dataset) - self.window_size and self.stride_compensation != "drop":
                    raise LookupError("Could not divide at least one of the given dataset into requested sequences. Last offset does not match end of the original dataset.")
                if i_dataset_idx_infos[-1]["sequence_end"] < (len(dataset) + added_padding): 
                    raise LookupError("Could not divide at least one of the given dataset into requested sequences. Last variation does not match dataset limits")
                if len(i_dataset_idx_infos) != i_projected_variations: 
                    raise LookupError("Could not divide at least one of the given dataset into requested sequences. Projected variation count does not match actual variation count.")

            # add all idx information generated from this dataset
            self.idx_lookup.extend(i_dataset_idx_infos)
            self.dataset_variations.append(len(i_dataset_idx_infos))
            self.dataset_paddings.append(added_padding)
        
        
    def generate_sequence_lookup(self, dataset_index, sequence_start, sequence_end, padding=None):
        return {"dataset_index": dataset_index, "sequence_start": sequence_start, "sequence_end": sequence_end, "padding": padding}


    def __getitem__(self, idx):
        # get corresponding sequence information for this idx
        idx_info = self.idx_lookup[idx]

        # get slice of data for this idx
        data_slice = self.datasets[idx_info["dataset_index"]][idx_info["sequence_start"]:idx_info["sequence_end"]]

        # add padding if needed
        if idx_info["padding"] != None and idx_info["padding"] > 0:
            data_slice = data_processing_util.add_padding(data_slice, idx_info["padding"], self.padding_dim)

        # return slice
        return data_slice
       

    def __len__(self):
        return len(self.idx_lookup)
        
        

        
