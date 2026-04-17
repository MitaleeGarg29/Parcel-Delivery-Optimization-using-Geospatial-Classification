import torch
import numpy as np
import pandas as pd
import requests
import os
import json
import random
import math

from datetime import date, time
from string import Template
from torch.utils.data import TensorDataset

from .crawler.date.load_holiday_data import get_date_frame
from .crawler.interest_point.load_ip_data import get_ip_frame


class artificial_data_util:

    """
    Provides functions to generate data for an artifial user. A user with the same ID and the same settings will always generate the same results.
    """
    @staticmethod
    def generate_user(user_id, start_date, start_time, end_date, end_time, timeslots_per_day, used_weekdays, max_regular_absences, max_home_timeslots, irregular_absences, generate_readable_fields) -> tuple:
        time_range = pd.date_range(start=str(start_date) + " " + str(start_time) , end=str(end_date) + " " + str(end_time), freq='15T', inclusive="both")

        i_dataframe = pd.DataFrame({'timestamp': time_range, "coordinate": float('NaN'), "lat": float('NaN'), "long": float('NaN'), "distance": float('NaN'), "copy": None})
        i_dataframe.sort_values(by='timestamp', inplace=True)
        i_dataframe.reset_index(drop=True, inplace=True)       

        # drop dates and times outside of desired time frames
        i_dataframe = i_dataframe[i_dataframe['timestamp'].dt.date <= date.fromisoformat(end_date)]
        i_dataframe = i_dataframe[i_dataframe['timestamp'].dt.date >= date.fromisoformat(start_date)]
        i_dataframe = i_dataframe[i_dataframe['timestamp'].dt.time <= time.fromisoformat(end_time)] 
        i_dataframe = i_dataframe[i_dataframe['timestamp'].dt.time >= time.fromisoformat(start_time)]
        i_dataframe = i_dataframe[i_dataframe['timestamp'].dt.weekday.isin(used_weekdays)] 

        i_dataframe.sort_values(by='timestamp', inplace=True)
        i_dataframe.reset_index(drop=True, inplace=True)     

        day_count = len(i_dataframe)/timeslots_per_day
        week_count = int(day_count/len(used_weekdays))

        # generate home location
        random.seed(user_id)
        i_user_lat = random.randint(0, 25000)
        random.seed(1000000-user_id)
        i_user_long = random.randint(0, 25000)
        i_user_home = [i_user_lat, i_user_long]

        # generate weekly movement patterns
        i_average_presence_duration = 0
        random.seed(user_id)
        i_external_locations_count = random.randint(2, max_regular_absences)
        i_external_location_timespans = int(((len(used_weekdays) * timeslots_per_day))/i_external_locations_count)

        last_set_time = None
        for week_index in range(week_count):
            week_start = week_index * len(used_weekdays) * timeslots_per_day

            i_average_presence_duration = 0
            for external_location_index in range(i_external_locations_count):
                location_seed = user_id + (1/(external_location_index + 1))
                external_location_time = week_start + (external_location_index * i_external_location_timespans)
                return_time = int(external_location_time + (i_external_location_timespans * 0.5))

                random.seed(location_seed)
                home_timeslots = random.randint(2, max_home_timeslots)

                random.seed(location_seed)
                external_location_lat = random.randint(0, 25000)
                random.seed(-location_seed)
                external_location_long = random.randint(0, 25000)

                # make user reach destination at beginning of absence time span
                i_dataframe.iloc[external_location_time, i_dataframe.columns.get_loc("lat")] = external_location_lat
                i_dataframe.iloc[external_location_time, i_dataframe.columns.get_loc("long")] = external_location_long

                # make user return home in the middle of the absence time span
                i_dataframe.iloc[return_time - int(home_timeslots/2) : return_time + int(home_timeslots/2), i_dataframe.columns.get_loc("lat")] = i_user_lat
                i_dataframe.iloc[return_time - int(home_timeslots/2) : return_time + int(home_timeslots/2), i_dataframe.columns.get_loc("long")] = i_user_long
                i_average_presence_duration += home_timeslots / i_external_locations_count
                last_set_time = return_time + int(home_timeslots/2)

        # set last partial week by copying previous week
        copy_amount = len(i_dataframe.iloc[last_set_time:])
        week_timeslots = timeslots_per_day * len(used_weekdays)
        i_dataframe.iloc[last_set_time:, i_dataframe.columns.get_loc("lat")] = i_dataframe.iloc[last_set_time - week_timeslots:last_set_time - week_timeslots + copy_amount, i_dataframe.columns.get_loc("lat")]
        i_dataframe.iloc[last_set_time:, i_dataframe.columns.get_loc("long")] = i_dataframe.iloc[last_set_time - week_timeslots:last_set_time - week_timeslots + copy_amount, i_dataframe.columns.get_loc("long")]
        i_dataframe.iloc[last_set_time:, i_dataframe.columns.get_loc("copy")] = "copied"

        # generate irregular absences
        i_irregular_absences_count = 0
        if irregular_absences:
            random.seed(user_id)
            i_irregular_absences_count = random.randint(1, day_count/30) # up to once per month; TODO: distribute over multiple months with empty months in between
            for absence_index in range(i_irregular_absences_count):
                absence_seed = user_id + (1/(absence_index + 1))
                random.seed(absence_seed)
                absence_duration = random.randint(3, 14)
                random.seed(absence_seed)
                absence_start = random.randint(absence_index*30, absence_index*30 + 16)

                random.seed(absence_seed)
                absence_lat = random.randint(0, 25000)
                random.seed(-absence_seed)
                absence_long = random.randint(0, 25000)

                i_dataframe.iloc[absence_start * timeslots_per_day:(absence_start + absence_duration) * timeslots_per_day, i_dataframe.columns.get_loc("lat")] = absence_lat
                i_dataframe.iloc[absence_start * timeslots_per_day:(absence_start + absence_duration) * timeslots_per_day, i_dataframe.columns.get_loc("long")] = absence_long

        # interpolate lat/long fields
        i_dataframe['lat'].interpolate("linear", inplace=True)
        i_dataframe['long'].interpolate("linear", inplace=True)
        i_dataframe = i_dataframe.round(0)

        # combine lat/long into coordinates
        i_dataframe["coordinate"] = i_dataframe.apply(lambda row: [int(row["lat"]), int(row["long"])], axis=1)

        # either exactly mimic real data or generate extra fields for readability
        if not generate_readable_fields:
            i_dataframe.drop('lat', axis=1, inplace=True)
            i_dataframe.drop('long', axis=1, inplace=True)
            i_dataframe.drop('distance', axis=1, inplace=True)
        else:
            i_dataframe["distance"] = i_dataframe.apply(lambda row: int(math.sqrt(pow(row["lat"] - i_user_lat, 2) + pow(row["long"] - i_user_long, 2))), axis=1)

        return i_user_home, i_dataframe, day_count, week_count, i_external_locations_count, i_irregular_absences_count, i_average_presence_duration
    
    
    @staticmethod
    def sanity_check_params(used_weekdays, timeslots_per_day, max_regular_absences, max_home_timeslots) -> bool:
        max_external_location_timespans = int(((len(used_weekdays) * timeslots_per_day))/max_regular_absences)
        if (max_external_location_timespans - max_home_timeslots) < 2: return False
        else: return True

    @staticmethod
    def dataframes_to_string(user_homes, user_dataframes, user_count) -> str:
        raw_data = "["
        for user_index_ in range(user_count):
            # add user home to the list
            raw_data += '{"coordinate": [' + str(user_homes[user_index_][0]) + "," + str(user_homes[user_index_][1]) + "]},"

            # add user trajectories to list
            raw_data += user_dataframes[user_index_].to_json(date_unit="ms", orient="records") + ","

        raw_data = raw_data[0:-1] + "]" 
        return raw_data



    

