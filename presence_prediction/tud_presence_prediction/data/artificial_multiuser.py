import random
import pandas as pd
import math
import json

from datetime import date, time
from torch.utils.data import TensorDataset

from .internal.data_procurer import data_procurer
from .internal.data_processing import data_processing_util
from .internal.artificial_data import artificial_data_util


class artificial_multiuser(data_procurer):
    """
    A data procurer subclass meant for analytic trainings and evaluations on artificial data.
    Deterministically generates data based on the settings given in the class variables.

    Will produce any chosen amount of users without storing them in the file system.
    Will reliably reproduce the same patterns for the same user ID.
    Each user will have an arbitrary amount of weekly absences and presences. 
    Each user as an arbitrary home address. 
    Each absence will lead the user to an arbitrarily generated external location.
    Each presence will last for an arbitrary amount of timeslots. 
    Between absences and presences, a user coordinates are interpolated linearily, simulating the users movement between locations.

    Uses the standard user data processing to generate input tensors containing coordinates, distances, weekdays and holidays, as well as label/ground truth tensors defining presence or absence. 
    
    Loads and processes a list of users, for each of which a list of training and validation data is returned, given in the form of TensorDatasets.
    """

    user_count = 2
    start_date = "2023-12-01"
    start_time = "08:00:00"
    end_date = "2024-03-23"
    end_time = "20:45:00"
    timeslots_per_day = 52
    used_weekdays = {0,1,2,3,4,5}
    max_regular_absences = 18           # Maximum of weekly external locations a user may visit. Exact number is determined at random for every user.
    max_home_timeslots = 14             # Maximum of how long the user will stay at home after each regular absence.
    irregular_absences = False          # If True, irregular absences are generated in addition to regular ones. Up to one per month, for between 3 and 14 days at a time, decided at random, somewhere within each 30 day window. 

    generate_readable_fields = True     # Generates additional fields that for manual analysis in CSV files

    def _load(self, **kwargs) -> str:
        # sanity check params
        if artificial_data_util.sanity_check_params(self.used_weekdays, self.timeslots_per_day, self.max_regular_absences, self.max_home_timeslots) == False:
            raise ValueError(f"Invalid parameters. User may not be able to be away that often while still being at home for {self.max_home_timeslots} time slots every time.")

        # log used settings
        self.logger.headline("Generating artifical users")
        self.logger.info("Settings:")
        self.logger.info(f" - User count: {self.user_count}")
        self.logger.info(f" - Start date: {self.start_date}")
        self.logger.info(f" - End date: {self.end_date}")
        self.logger.info(f" - Start time: {self.start_time}")
        self.logger.info(f" - End time: {self.end_time}")
        self.logger.info(f" - Timeslots: {self.timeslots_per_day}")
        self.logger.info(f" - Used weekdays: {self.used_weekdays}")
        self.logger.info(f" - Maximum weekly absences: {self.max_regular_absences}")
        self.logger.info(f" - Maximum home timeslots per absence: {self.max_home_timeslots}")
        self.logger.info(f" - Irregular absences: {self.irregular_absences}")

        user_homes = []
        user_dataframes = []
        for user_index in range(self.user_count):
            # generate data
            i_user_home, i_user_data, i_day_count, i_week_count, i_external_locations_count, i_irregular_absences_count, i_average_presence_duration = artificial_data_util.generate_user(user_index, self.start_date, self.start_time, self.end_date, self.end_time, self.timeslots_per_day, self.used_weekdays, self.max_regular_absences, self.max_home_timeslots, self.irregular_absences, self.generate_readable_fields)

            # save user
            user_homes.append(i_user_home)
            user_dataframes.append(i_user_data)

            # log user stats
            self.logger.info(f"Stats for artifical user {user_index}", True)
            self.logger.info(f" - Home address: {user_homes[user_index]}")
            self.logger.info(f" - Generated timespan: {i_day_count} days/{i_week_count} weeks")
            self.logger.info(f" - Weekly absences: {i_external_locations_count}")
            self.logger.info(f" - Average continous presence: {i_average_presence_duration} time slots")
            self.logger.info(f" - Irregular absences: {i_irregular_absences_count}")

        
        # transform to string to imitate real raw format
        raw_data = artificial_data_util.dataframes_to_string(user_homes, user_dataframes, self.user_count)

        return raw_data

    
    def _process(self, loaded_raw_data, **kwargs) -> list[list[TensorDataset]]:
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
