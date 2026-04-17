import json
import argparse
import datetime

from torch.utils.data import TensorDataset

from .internal.data_procurer import data_procurer
from .internal.data_processing import data_processing_util


class dynamic_multiuser(data_procurer):
    """
    A data procurer subclass meant for productive trainings and evaluations.

    Procures data for all users defined in a local file or environment variable from a HTTP source.
    Uses the standard user data processing to generate input tensors containing coordinates, distances, weekdays and holidays, as well as label/ground truth tensors defining presence or absence. 
    
    Loads and processes a list of users, for each of which a list of training and validation data is returned, given in the form of TensorDatasets.
    """
    def _load(self, **kwargs) -> str:
        # get user's IDs
        user_ids = data_processing_util.get_all_users()
        self.logger.info(f"Retrieved user URL components for {len(user_ids)} users.")

        # get raw data
        raw_data = "["
        for index, row in user_ids.iterrows():
            i_user_data = data_processing_util.load_user_data(row[user_ids.columns[1]], row[user_ids.columns[2]], row[user_ids.columns[3]], row[user_ids.columns[4]])
            i_user_data = i_user_data[1:-1] + "," # drop last bracket in json syntax to join user lists together
            raw_data += i_user_data

            self.logger.info(f"Retrieved data for user {index}.")
        raw_data = raw_data[0:-1] + "]" # drop last comma and close list
        self.logger.info("Retrieved user data from URLs.")

        return raw_data

    
    def _process(self, loaded_raw_data, **kwargs) -> list[tuple[TensorDataset]]:
        # load raw data from JSON string
        json_data = json.loads(loaded_raw_data)
        
        all_datasets = []
        for index in range(0, len(json_data), 2):
            # split raw data
            home_coordinate = json_data[index]['coordinate']
            user_data = json_data[index + 1]
            
            extra_days = None
            if kwargs.get("extra", None) != None: extra_days = kwargs.get("days", None) - 1 if kwargs.get("days", None) != None else 0
            # set extrapolation (used for predictions)
            extrapolation = kwargs.get("extrapolate", None)

            # process data
            user_datasets = data_processing_util.process_user(home_coordinate, user_data, min_date=None, max_date=None, extra_days=extra_days, extrapolate_to=extrapolation, validation_split_percentage=kwargs.get("split", None), count_interpolation=True)
            all_datasets.append(user_datasets)

        return all_datasets
    
    
    def _set_data_headers(self):
        data_headers = [
            #["Lat", "Long", "From Home", "From POI 1", "From POI 2", "From POI 3", "From POI 4", "From POI 5", "From POI 6", "From POI 7", "From POI 8", "FromPOI 9", "From POI 10"],
            ["Lat", "Long", "Home-Distance", "SD-Home-Distance"],
            ["Not Holiday", "Is Holiday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Timeslot"],
            ["Label"]
        ]

        return data_headers
