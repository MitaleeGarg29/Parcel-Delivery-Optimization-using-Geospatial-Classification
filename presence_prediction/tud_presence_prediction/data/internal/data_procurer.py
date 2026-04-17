import json
import torch
from datetime import datetime
from sys import getsizeof
from os import path, listdir
from torch.utils.data import TensorDataset

import pandas as pd
import numpy as np

from .data_processing import data_processing_util


class data_procurer: 
    """
    A data_procurer represents a standardized way to procure data for this project. This base class implements functionality used by all data_procurer subclasses but should never be used directly.
    
    The functions "_load()", "_process()" and optionally "_set_data_headers()" should be overwritten by all subclasses. 
    Their respective documentation contains information about the desired return values.

    A subclass additionally inherits methods from this base class which will be used externally. 
    For example, every subclass will automatically offer the "load()" method implemented within this class which internally calls _load(), but also takes care of using the offline storage features and logging.

    Additionally, this class offers automatic time-shifting functionality and offline storage features for all properly defined subclasses adhering to the defined standards.
    """

    local_data_directory = "data" + path.sep + "local_data"
    raw_filename_suffix = "raw"
    processed_filename_suffix = "processed"
    storage_filename_extension = ".txt"
    readable_filename_extension = ".csv"
    training_data_additional_suffix = "training"
    validation_data_additional_suffix = "validation"
    storage_directory_name = "storage"
    readable_directory_name = "readable"

    def __init__(self, logger, from_storage, to_storage, storage_file="auto", store_readable=True):
        logger.info("Initializing data procurer.")

        self.raw_data = None
        self.datasets = None
        self.logger = logger
        self.from_storage = from_storage
        self.to_storage = to_storage
        self.store_readable = store_readable

        self.base_filename = ""
        self.filename_prefix = self.__class__.__name__
        self.base_filename = ""
        self.raw_storage_file = ""
        self.raw_readable_file = ""
        self.processed_storage_file = "" # placeholder
        self.processed_readable_file = ""


        if (self.to_storage or self.from_storage) and (storage_file == None or len(storage_file) < 1): 
            raise ValueError("No storage file name provided.")
        
        if self.to_storage and self.from_storage:
            raise ValueError("Can not save to storage and load from storage at the same time.")
        
        # resolve 'auto' storage file name
        if storage_file == "auto": 
            if from_storage == True:
                all_auto_gen_files = [filename for filename in listdir(self.get_full_file_path(is_storage=True)) if filename.endswith(self.storage_filename_extension) and filename.startswith(self.filename_prefix)]
                logger.info(f"Found {len(all_auto_gen_files)} potential local data files.")
                if len(all_auto_gen_files) == 0:
                    self.base_filename = "None found"
                else:
                    all_auto_gen_files.sort()
                    found_filename = all_auto_gen_files[-1]
                    self.base_filename = self.extract_base_filename(found_filename)
                    self.raw_storage_file = self.generate_full_filename(self.base_filename, True, True)
                    
            elif to_storage == True or store_readable == True:
                self.base_filename = self.dt_to_base_filename(datetime.now())
                self.raw_storage_file = self.generate_full_filename(self.base_filename, True, True)
        # use given filename as base name and as raw storage name
        else:
            if not storage_file.endswith(self.storage_filename_extension): 
                self.base_filename = storage_file + self.dt_to_base_filename(datetime.now())
                self.raw_storage_file = storage_file + self.storage_filename_extension
            else: 
                self.base_filename = storage_file[0:-(len(self.storage_filename_extension))] + self.dt_to_base_filename(datetime.now())
                self.raw_storage_file = storage_file

        if self.from_storage and not path.isfile(self.get_full_file_path(self.get_full_filename(True, True), is_storage=True)):
            raise ValueError("The file to load data from has not been found.")
            #self.logger.critical(f"The file to load data from has not been found. Falling back to live procurement.")
            #self.base_filename = self.dt_to_base_filename(datetime.now())
            #self.raw_storage_file = self.get_full_filename(True, True)
            #self.from_storage = False
 
        # generate missing file names for other types of local files
        self.raw_readable_file = self.generate_full_filename(self.base_filename, True, False)
        self.processed_readable_file = self.generate_full_filename(self.base_filename, False, False)

    
    def load(self, shift = None, **kwargs):
        self.logger.info(f"Data procurer {self.__class__.__name__} starts loading data from {self.get_source_name()}.")
    
        # load raw data
        if self.from_storage:
            self.raw_data = self.load_raw_data_from_storage_file()
        else:
            self.raw_data = self._load(**kwargs)

        if self.raw_data == "" or self.raw_data == None: raise ValueError("Data could not be loaded.")
        
        self.raw_data_size = self.get_raw_data_size()
       
        self.logger.info(f"{self.raw_data_size} Bytes of raw data loaded succesfully from {self.get_source_name()}.")

        # save raw data to storage
        if self.to_storage and not self.from_storage:
            self.save_raw_data_to_storage_file()
            self.logger.info(f"Saved raw data to file {self.raw_storage_file}.")
        # save raw data in readable format
        if self.store_readable: # raw data is unchanged if loaded from storage
            raw_data_readable = self.save_raw_data_to_readable_file()
            if raw_data_readable: self.logger.info(f"Saved readable raw data to file {self.raw_readable_file}.")

        # create processed data
        self.logger.info(f"Processing data.")
        self.datasets = self.process_raw_data(shift, **kwargs)
        self.datasets_size = self.get_dataset_size()
        self.logger.info(f"Aquired {self.datasets_size} Bytes of processed data.")

        # save processed data
        if self.store_readable:
            processed_data_readable = self.save_processed_data_to_readable_file()
            if processed_data_readable: self.logger.info(f"Saved readable processed data to file {self.processed_readable_file}.")

        return self.datasets
    
    def process_raw_data(self, shift = None, **kwargs):
        # let subclass perform its preprocessing
        processed_data_sets = self._process(self.raw_data, **kwargs)

        shifted_data_sets = []
        # check if data procurer subclass returned data in the correct format
        if type(processed_data_sets) == list and len(processed_data_sets) == 2 and isinstance(processed_data_sets[0], TensorDataset) and isinstance(processed_data_sets[1], type(None)):
            # data procurer returned list[TensorDataset, TensorDataset]
            shifted_data_sets.append(processed_data_sets)
        elif type(processed_data_sets) == list and all(len(innerList) == 2 and isinstance(innerList, list) and isinstance(innerList[0], TensorDataset) and isinstance(innerList[1], (TensorDataset, type(None))) for innerList in processed_data_sets):
            # data procurer returned list[list[TensorDataset, TensorDataset]]
            shifted_data_sets = processed_data_sets
        else:
            raise TypeError("The data procurer has returned invalid data and is likely not implemented correctly. A data procurer's data processing must return a single list or a list of lists. These lists must contain a data set for training and optionally for validation.")

        for processed_list in shifted_data_sets:
            for set in processed_list: 
                if set != None and None in data_processing_util.get_dataset_metadata(set).values():
                   self.logger.critical("The data procurer did not generate full metadata.")

        # shift data
        shift = data_processing_util.interpret_shift_string(shift)
        if shift != None and shift != 0:
            for i_user_data in shifted_data_sets:
                for i_user_set in range(2): # training and validation
                    if i_user_data[i_user_set] != None:
                        shift_dims = 1
                        if len(i_user_data[i_user_set].tensors[0].shape) > 2: shift_dims = 2
                        # assume first tensor contains future unknowns (the type of input data that must be shifted) and shift its contents according to shift param
                        # TODO: If neccessary, make sure this actually shifts time slots onto previous days (likely needs to be flattened, which should happen when using different params for roll)
                        # additionally remove first rows from all tensors as the input data is now incomplete
                        new_tensors = []
                        for i_user_tensor_num in range(len(i_user_data[i_user_set].tensors)):
                            if shift_dims >= 2:
                                if i_user_tensor_num == 0: 
                                    new_tensors.append(i_user_data[i_user_set].tensors[i_user_tensor_num].roll(shift, (0, 1))[shift[0]:,shift[1]:])
                                else:
                                    new_tensors.append(i_user_data[i_user_set].tensors[i_user_tensor_num][shift[0]:,shift[1]:])
                            else:
                                if i_user_tensor_num == 0:
                                    new_tensors.append(i_user_data[i_user_set].tensors[i_user_tensor_num].roll(shift[0], 0)[shift[0]:])    
                                else:
                                    new_tensors.append(i_user_data[i_user_set].tensors[i_user_tensor_num][shift[0]:])
                        i_user_data[i_user_set].tensors = tuple(new_tensors)


        return shifted_data_sets


    def get_raw_data_size(self):
        raw_data_size = None
        try:
            raw_data_size = len(self.raw_data.encode("utf-8"))
        except Exception as e:
            self.logger.info(f"Could not read size of raw data aquired by {self.__class__.__name__} due to exception: {e}.")
            raw_data_size = "Unknown"
        return raw_data_size
    
    def get_dataset_size(self):
        datasets_size = None
        try:
            datasets_size = 0
            for i_user_data in self.datasets:
                for training_tensor in i_user_data[0].tensors:
                    datasets_size += self.get_tensor_data_size(training_tensor)
                if len(self.datasets) > 1 and i_user_data[1] != None:
                    for validation_tensor in i_user_data[1].tensors:
                        datasets_size += self.get_tensor_data_size(validation_tensor)
        except Exception as e:
            self.logger.critical(f"Could not read size of processed data aquired by {self.__class__.__name__} due to exception: {e}.")
            datasets_size = "Unknown"
        return datasets_size
    
    def get_tensor_data_size(self, tensor_data):
        return getsizeof(tensor_data) + tensor_data.nelement() * tensor_data.element_size()

    def get_source_name(self):
        source_name = f"local storage ({self.raw_storage_file})" if self.from_storage else "live source" 
        return source_name
    
    # --- file operations ---
    def dt_to_base_filename(self, datetime):
        datetime_string = datetime.isoformat().replace(",", "").replace(":", "").replace(".", "").replace("-", "")
        return datetime_string
    
    def get_full_filename(self, is_raw, is_storage):
        if is_raw and is_storage: 
            return self.raw_storage_file
        elif is_raw and not is_storage:
            return self.raw_readable_file
        elif not is_raw:
            return self.processed_storage_file

    def generate_full_filename(self, base_name, is_raw, is_storage, is_training=None, append_additionally=None):
        full_filename = self.filename_prefix + "_" + base_name + "_"
        
        # add raw/processed suffix
        if is_raw: 
            full_filename += self.raw_filename_suffix
        else:
            full_filename += self.processed_filename_suffix

        # optionally add distinction between training data and validation data
        if is_training == True:
            full_filename += "_" + self.training_data_additional_suffix
        elif is_training == False:
            full_filename += "_" + self.validation_data_additional_suffix

        # optionally add arbitrary string
        if append_additionally != None:
            full_filename += "_" + append_additionally

        # add extension
        if is_storage:
            full_filename += self.storage_filename_extension
        else: 
            full_filename += self.readable_filename_extension 

        return full_filename
    
    def extract_base_filename(self, full_filename):
        split_filename = full_filename.split("_")
        # check if file matches auto generated file names. If so, base name can be deducted from that. Otherwise use full name as new base name.
        if len(split_filename) > 1 and self.generate_full_filename(split_filename[-2], True, True) == full_filename:
            return split_filename[-2] # contains base name for auto generated file names
        else: return full_filename
        
    def get_full_file_path(self, filename=None, is_storage=None):
        sub_directory = None
        if is_storage == True: sub_directory = self.storage_directory_name
        elif is_storage == False: sub_directory = self.readable_directory_name
        else: raise ValueError("Need sub-directory.")

        # return path to file if file is given, otherwise just return directory
        if not filename or filename == None or filename == "": return path.join(self.local_data_directory, sub_directory)
        else: return path.join(self.local_data_directory, sub_directory, filename)



    def save_to_file(self, full_file_path, content):
        with open(full_file_path, 'w+', encoding="UTF-8") as content_file:
            content_file.write(content)

    def load_from_file(self, full_file_path):
        with open(full_file_path, 'r') as content_file:
            file_content = content_file.read()
        return file_content

    def save_raw_data_to_storage_file(self):
        full_file_path = self.get_full_file_path(self.get_full_filename(True, True), is_storage=True)
        self.save_to_file(full_file_path, self.raw_data)
        

    def load_raw_data_from_storage_file(self):
        full_file_path = self.get_full_file_path(self.get_full_filename(True, True), is_storage=True)
        return self.load_from_file(full_file_path)

    
    def save_raw_data_to_readable_file(self):
        full_file_path = self.get_full_file_path(self.get_full_filename(True, False), is_storage=False)

        json_data = None
        try: 
            json_data = json.loads(self.raw_data)
        except: 
            self.logger.critical("Could not save raw data to readable file since data could not be parsed by JSON-parser.")
            json_data = None
            return False
        
        full_csv_text_data = ""

        first_dim_type = "Unknown"
        if isinstance(json_data, dict): first_dim_type = "dict"
        if isinstance(json_data, list): first_dim_type = "list"
            
        if first_dim_type == "Unknown":
            self.logger.critical("Could not save raw data to readable file since JSON data had unexpected structure.")
            return False      
        elif len(json_data) < 1:
            self.logger.critical("Could not save raw data to readable file since JSON data was empty.")
            return False   
        else:
            try:
                all_pd_dataframes = []
                # parse all parts of list individually
                if first_dim_type == "list":
                    for list_item in json_data:
                        try:
                            all_pd_dataframes.append(pd.json_normalize(list_item, errors="ignore"))
                        except:
                            self.logger.critical(f"At least one of the transmitted JSON tables could not be parsed into CSV format.")
                            return False   
                # otherwise just assume object is normalizable already
                else:
                    all_pd_dataframes.append(pd.json_normalize(json_data, errors="ignore"))

                full_csv_text_data = "sep=," + "\n"

                for i_dataframe in all_pd_dataframes:
                    date_column = None
                    for i_column in i_dataframe:
                        if str(i_column).count("date") > 0 or str(i_column).count("time") > 0:
                            date_column = i_column
                            i_dataframe[i_column] = pd.to_datetime(i_dataframe[i_column], unit="ms")
                    # sort every DF by first date column found
                    #if date_column != None: i_dataframe.sort_values(by=date_column)
                    full_csv_text_data += i_dataframe.to_csv(index=False, line_terminator="\n") + "\n" + "\n" + "\n"

                self.save_to_file(full_file_path, full_csv_text_data)
                return True
            except Exception as e:
                self.logger.critical(f"Could not save raw data to readable file since JSON data had structure that is incompatible with 2D tables: {e}")
                return False   
            

    def save_processed_data_to_readable_file(self):
        # get columns names from sub-class, if implemented
        column_names = self._set_data_headers()
        named_tensors = len(column_names)
        if column_names is not None and named_tensors > 0 and isinstance(column_names[0], list):
            use_column_names = True
        else:
            self.logger.info("Data procurer did not provide valid column names for processed data. CSV-file will be saved without column headers.")

        for user_index, i_user_dataset in enumerate(self.datasets):

            # try to get 2D representation of final processed DataSets returned by sub class
            for dataset_type in ["training", "validation"]:
                if dataset_type == "validation" and (len(i_user_dataset) < 2 or i_user_dataset[1] == None): continue
                
                for index, dt_tensor in enumerate(i_user_dataset[0 if dataset_type == "training" else 1].tensors):
                    full_training_csv_text_data = "sep=," + "\n"

                    #training_tensor_flat = torch.flatten(dt_tensor, len(dt_tensor.shape) - 2, len(dt_tensor.shape) - 1) # remove last dimension    
                    training_tensor_flat = torch.flatten(dt_tensor, 0, len(dt_tensor.shape) - 2) if len(dt_tensor.shape) > 2 else dt_tensor # remove first dimension(s) until tensor is 2D

                    # try deducting desired 2D target shape by given headers 
                    used_column_axis = None
                    used_column_header = None
                    if named_tensors > index: 
                        headers_fit = False
                        # TODO (optional): Make more dynamic by looking for the right dimension before reshaping. Dimension with correct length should exist from beginning.  
                        # see if one of the two remaining dims fits the desired headers
                        for dim_index, dim_length in enumerate(training_tensor_flat.shape):
                            if dim_length == len(column_names[index]):
                                headers_fit = True
                                used_column_header = column_names[index]
                                used_column_axis = 1 - dim_index
                        
                        # if it doesn't fit yet, try to flatten further
                        if headers_fit == False and len(column_names[index]) == 1:        
                            training_tensor_flat = torch.flatten(training_tensor_flat)
                            headers_fit = True
                            used_column_header = column_names[index]
                            used_column_axis = 0

                        if headers_fit == False:
                            self.logger.info("The data provided by the data procurer could not be shaped to match the provided headers for the data. CSV file will be created without headers.")

                    training_np = training_tensor_flat.numpy()
                    training_df = pd.DataFrame(training_np)
                    if used_column_header is not None:
                        # TODO: test if transpose works
                        if used_column_axis == 1: 
                            training_df.reset_index(inplace=True)
                            training_df = training_df.transpose()
                        training_df.columns = used_column_header
                    
                    full_training_csv_text_data += training_df.to_csv(index=False, line_terminator="\n") + "\n" + "\n" + "\n"

                    tensor_type_name = f"set{user_index}_"
                    tensor_type_name += "input" + str(index) if (index + 1) < len(i_user_dataset[0 if dataset_type == "training" else 1].tensors) else "label"
                    
                    full_file_path = self.get_full_file_path(self.generate_full_filename(self.base_filename, False, False, dataset_type == "training", append_additionally=tensor_type_name), is_storage=False)
                    self.save_to_file(full_file_path, full_training_csv_text_data)


        return True


    


    def _load(self, **kwargs) -> str: 
        """
        This function must be implemented by every data_procurer sub-class. It must retrieve raw data in a string format. 
        This data may automatically be stored and retrieved locally depending on the data procurers arguments.
        If the string is in JSON format, an additional CSV-file may be created for manual inspection upon every new live procurement depending on the data procurers arguments.
        """
        raise NotImplementedError("The specified data procurer does not implement the neccessary function '_load'.")
    

    
    def _process(self, loaded_raw_data, **kwargs) -> list[list[TensorDataset, TensorDataset]]: 
        """
        This function must be implemented by every data_procurer sub-class. It must process the given string data. 
        The training and validation data must each be given as TensorDatasets. They must be returned as a list containing both elements.
        Optionally, that list may be contained in another list to utilize multiple datasets that will be used sequentially for training within one epoch.

        The last tensor in every TensorDataSets is presumed to be the target output. Pass a Tensor filled with NaN values if there are no labels available.
        All previous tensors are presumed to be the input data for the model. Up to three inputs are usable.
        The first among the input tensors is presumed to be a future unknown time series. If a shift parameter is given when loading the data, the data procurer will automatically shift this tensor.
        The Datasets should additionally provide metadata regarding the contained information. Refer to the function append_dataset_metadata(...) in the data_processing_util for information about required meta data.
        If the data_procurer only provides data for predictions, the validation DataSet is not used.
        If one does not want to provide validation data in general, the validation set may be replaced with None.
        """
        raise NotImplementedError("The specified data procurer does not implement the neccessary function '_process'.")
    

    def _set_data_headers(self) -> list[list[str]]:
        """
        This function may optionally be implemented to describe the data contained in the DataSets provided by '_process()'. 
        It is exclusively used to create .CSV files for human readability. 
        Column headers for the CSV file as well as the desired 2D shape of the data is deducted from the functions return values. 
        """
        return None
    
    
