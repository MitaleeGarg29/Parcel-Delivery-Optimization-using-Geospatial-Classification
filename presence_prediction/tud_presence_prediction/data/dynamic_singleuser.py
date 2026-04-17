import json
import argparse
import datetime

from torch.utils.data import TensorDataset

from .internal.data_procurer import data_procurer
from .internal.data_processing import data_processing_util


class dynamic_singleuser(data_procurer):
    """
    A data procurer subclass meant for productive predictions.

    Procures data for a given user. The raw data is either given as an argument to the _load function or is retrieved from an HTTP source using the users ID and URL information defined in a local file or environment variable. 
    Uses the standard user data processing to generate input tensors containing coordinates, distances, weekdays and holidays, as well as label/ground truth tensors defining presence or absence. 
    
    Loads and processes a users training and validation data, given in the form of TensorDatasets.
    """
    def _load(self, **kwargs) -> str:
        raw_data = None
        # get data directly from arguments
        if kwargs.get("user_home_coordinates", None) != None and kwargs.get("user_data", None):
            raw_data = "[" +  kwargs["user_home_coordinates"] + "," + kwargs["user_data"] + "]"
        # get data from HTTP source
        else:
            # get users internal ID
            user_internal_id = kwargs['user']
            self.logger.info(f"Retrieving IDs for user {user_internal_id}.")

            # get user's URL IDs
            user_ids = data_processing_util.get_user_by_id(user_internal_id)
            self.logger.info(f"Retrieved user URL components: {user_ids.iloc[0][user_ids.columns[1]]}, {user_ids.iloc[0][user_ids.columns[2]]}.")

            # get raw data
            raw_data = data_processing_util.load_user_data(user_ids.iloc[0][user_ids.columns[1]], user_ids.iloc[0][user_ids.columns[2]])
            self.logger.info("Retrieved user data from URL.")

        return raw_data

    
    def _process(self, loaded_raw_data, **kwargs) -> list[tuple[TensorDataset]]:
        # load raw data from JSON string
        json_data = json.loads(loaded_raw_data)
        
        # split raw data
        home_coordinate = json_data[0]['coordinate']
        user_data = json_data[1]

        # set latest date (used for testing only)
        prediction_end_date = None
        if kwargs.get("date", None) != None: prediction_end_date = datetime.date.fromisoformat(kwargs["date"])
        # procure extra dates (used for shift-based models only)
        extra_days = None
        if kwargs.get("extra", None) != None: extra_days = kwargs.get("days", None) - 1 if kwargs.get("days", None) != None else 0
        # set extrapolation (used for predictions)
        extrapolation = kwargs.get("extrapolate", None)

        # process data
        training_data, validation_data = data_processing_util.process_user(home_coordinate, user_data, min_date=None, max_date=prediction_end_date, extra_days=extra_days, extrapolate_to=extrapolation, validation_split_percentage=kwargs.get("split", None))
        return [training_data, validation_data]
    
    def _set_data_headers(self):
        data_headers = [
            #["Lat", "Long", "From Home", "From POI 1", "From POI 2", "From POI 3", "From POI 4", "From POI 5", "From POI 6", "From POI 7", "From POI 8", "FromPOI 9", "From POI 10"],
            ["Lat", "Long", "Home-Distance", "SD-Home-Distance"],
            ["Not Holiday", "Is Holiday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Timeslot"],
            ["Label"]
        ]

        return data_headers
