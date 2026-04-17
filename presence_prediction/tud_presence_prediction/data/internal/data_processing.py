import torch
import numpy as np
import pandas as pd
import requests
import os
import json
import datetime
import pkg_resources
import math

from string import Template
from torch.utils.data import TensorDataset
from sklearn.metrics import precision_score, accuracy_score

from .crawler.date.load_holiday_data import get_date_frame
from .crawler.interest_point.load_ip_data import get_ip_frame



class data_processing_util:
    """
    Offers a variety of data-related functionality.

    Data procurers may use methods provided by this class to load and process user data. 
    The process_user function offers a large amount of parameters to be used by different data procurers.

    Additionally, the general functionality to be used from any point within the project is provided, such as standardized padding generation or analyzing and logging dataset metrics.
    
    Class variables define specific information neccessary for this class to function properly. This information is ultimately defined by the productive environment and may change in the future.
    """


    test_mode = False
    user_id_columns = ["internal_user_id", "webhook_user_id", "webhook_home_id", "webhook_start_date", "webhook_end_date"]
    user_file_path = pkg_resources.resource_filename("tud_presence_prediction","data/users/ids.txt")
    env_var_name= "user_ids"
    urls = [f'https://rl.green-convenience.com/api/v1/positions/convert/${user_id_columns[1]}/${user_id_columns[2]}',
            f"https://rl.green-convenience.com/api/v1/positions/preprocess/${user_id_columns[1]}/${user_id_columns[2]}/"]
    
    padding_value = -1

    @staticmethod
    def get_all_users():
        """
        Returns a list of all available user IDs and URL information. User IDs will be retrieved from the environment variable defined above. 
        If that variable is empty, a local file containing user information is used instead.
        User ID's can be in any format, but must identify the user uniquely and permanently.
        """
        # if no user IDs are found, replace them by reading from file system
        if os.environ.get(data_processing_util.env_var_name) == None:
            all_users_from_file = pd.read_csv(data_processing_util.user_file_path, names=data_processing_util.user_id_columns, comment="#")
            os.environ[data_processing_util.env_var_name] = all_users_from_file.to_json()

        #all_users = pd.json_normalize(json.loads(os.environ['user_ids']))
        all_users = pd.DataFrame(json.loads(os.environ[data_processing_util.env_var_name]))
    
        return all_users

    @staticmethod
    def get_user_by_id(id):
        """
        Returns a list of all available user IDs and URL information. User IDs will be retrieved from the environment variable defined above. 
        If that variable is empty, a local file containing user information is used instead.
        User ID's can be in any format, but must identify the user uniquely and permanently.
        """
        all_users = data_processing_util.get_all_users()
        found_user = all_users.loc[all_users[data_processing_util.user_id_columns[0]] == id]

        if len(found_user) != 1:
            raise ValueError("Error occured while trying to retrieve user for given ID. Either the given user is not present or the collection of users is malformed.")

        return found_user
    
    @staticmethod
    def load_user_data(user_id, home_id, start_date, end_date):
        """
        Requests user data from a HTTP source. Base URL and parameters are defined by the user and must be unique.
        """
        # Get data from database
        text_data = "["
        for index, url in enumerate(data_processing_util.urls):
            url_params = {data_processing_util.user_id_columns[1]: user_id, data_processing_util.user_id_columns[2]: home_id}
            url = Template(url).safe_substitute(url_params)
            query_params = []
            if index == 1:
                if start_date:
                    query_params.append(f"start_date={start_date}")

                if end_date:
                    query_params.append(f"end_date={end_date}")

                if query_params:
                    url += "?" + "&".join(query_params)
                    
            response = requests.get(url)

            # Check if the request was successful
            if response.status_code == 200:
                text_data += response.text + ("," if index < (len(data_processing_util.urls) -1) else "")
            else:
                raise RuntimeError(f"Failed to retrieve data from the URL. Status code: {response.status_code}")
                return None

        text_data += "]"    

        return text_data



    @staticmethod
    def process_user(home_coordinate, user_data, min_time="08:00:00", max_time="20:45:00", time_steps=15, home_cutoff=50, interpolation='ffill', extrapolate_to="now", week_days={0,1,2,3,4,5}, min_date=None, max_date=None, extra_days=0, validation_split_percentage=0, count_interpolation=False, input_replacement_value="auto", generate_location=True, generate_weather=False, normalize=True, add_timeslot_feature=True, add_temporal_features=False, cluster_count=0) -> tuple[TensorDataset]:
        r"""Processes a given users data. Expects at least the users home coordinates as a JSON key-value-pair and the users relative locations in JSON list of key-value-pairs. Returns a training set and optionally a validation set of processed data.
    
        Args:
            home_coordinate (string): 
                The users home coordinates as a string for latitude and longitude.
            user_data (list[dict]): 
                A list of dicts providing 'coordinate' and 'timestamp' keys for a sequence of the users relative locations.
            min_time (time or string): 
                Defines the beginning of the daily schedule, which is enforced for every day. Missing data is interpolated/extrapolated and surplus data is discarded. This should not include times that are never present in the input data.
            max_time (time or string): 
                Defines the beginning of the daily schedule, which is enforced for every day. Missing data is interpolated/extrapolated and surplus data is discarded. This should not include times that are never present in the input data.
            time_steps (int):
                Defines the time between data points. Must match the input data.
            home_cutoff: 
                Defines at which relative distance from his home the user is considered to be at home.
            interpolation: 
                Defines the interpolation method to be used. Supports 'ffill' (forward fill), 'linear' and None.
            extrapolate_to: 
                Defines up to which point the location data should be extrapolated. Supports 'now' to extrapolate to current datetime, 'last' to extrapolate to the last day and time in the set and None. Uses the same method that is used for interpolation and requires interpolation to be active.
            week_days:
                A zero based list of week-day indices. The given weekdays are included in the dataset, the remaining ones are dropped. This should not include days that are missing in the input data.
            min_date (date or string): 
                The output will be truncated at the given date. Does not affect interpolation/extrapolation. This may be used for effiency in some cases or for testing purposes. 
            max_date (date or string): 
                The output will be truncated at the given date. Does not affect interpolation/extrapolation. This may be used for effiency in some cases or for testing purposes.
            extra_days (int):
                The output will be lengthend by the given amount of days. This can be used to add additional 'future known' information, which some models may use. 
            validation_split_percentage (float): 
                Float number between 0 and 99 which indicates how much of the labeled data should be reserved for the validation set. If set to 0, then no validation set is created and all data is used for the training set.
            count_interpolation (bool)
                If set to True, the amount of required interpolation is calculated at the expense of computational cost.
            input_replacement_value (any):
                Determines the value that should be inserted when values are missing. If set to "auto", values will be replaced with the standard padding value (recommended). Does not apply to the label tensor as missing label values are always replaced with nan values.
            normalize (bool):
                Wether location data should be normalized.
            add_timeslot_feature (bool):
                Wether a continous number representing the time of the day, relative to the used earliest and latest time, should be added as a feature.
            add_temporal_features (bool):
                If True, additional lagged distances are generated.
            generate_location: 
                (Not implemented yet). Wether location data should be included in the final dataset. Can be used to train a model that only captures seasonality, which does not depend on current location data at prediction time.
            generate_weather:
                (Not implemented yet). Wether weather data should be included in the final dataset.
            cluster_count: 
                (Not implemented). A number higher than 0 will create additional data by automatically calculating cluster centers and distances to these centers at every time step. This is currently not supported as it introduces vast amounts of randomness during prediction time.

        Returns:
            [TensorDataset, TensorDataset]: Returns a training TensorDataset and optionally a validation TensorDataset. If there is no validation split, the latter is replaced with None.
        """
        
        # process params
        if type(max_date) == str: max_date = datetime.date.fromisoformat(max_date)
        elif type(max_date) == datetime.datetime: max_date = max_date.date()
        elif type(max_date) != datetime.date and max_date != None: raise ValueError("Invalid date format.")

        if type(min_date) == str: min_date = datetime.date.fromisoformat(min_date)
        elif type(min_date) == datetime.datetime: min_date = min_date.date()
        elif type(min_date) != datetime.date and min_date != None: raise ValueError("Invalid date format.")

        if type(max_time) == str: max_time = datetime.time.fromisoformat(max_time)
        elif type(max_time) == datetime.datetime: max_time = max_time.time()
        elif max_time == None: max_time = datetime.time.fromisoformat("23:59:00")
        elif type(max_time) != datetime.time: raise ValueError("Invalid time format.")
        
        if type(min_time) == str: min_time = datetime.time.fromisoformat(min_time)
        elif type(min_time) == datetime.datetime: min_time = min_time.time()
        elif min_time == None: min_time = datetime.time.fromisoformat("00:00:00")
        elif type(min_time) != datetime.time: raise ValueError("Invalid time format.")

        if validation_split_percentage == None: validation_split_percentage = 0

        if input_replacement_value == "auto": input_replacement_value = data_processing_util.padding_value

        if generate_weather == True: raise NotImplementedError("Weather data is currently not supported.")
        if cluster_count > 0: raise NotImplementedError("Clustering of POIs is currently not supported as the amount of randomness induced by it represents too much noise.")
        if interpolation != None and interpolation != "ffill" and interpolation != "linear": raise ValueError("Only 'ffill' or 'linear' interpolation or no interpolation is supported.")
        if type(week_days) != set or max(week_days) > 6 or min(week_days) < 0: raise ValueError("Invalid set of week days.")
        if type(validation_split_percentage) != int or validation_split_percentage > 99 or validation_split_percentage < 0: raise ValueError("Validation split percentage must be between 0 and 99")
        if interpolation != None and type(extrapolate_to) != str and extrapolate_to not in ("now", "last", "none"): raise ValueError("Extrapolation supports variants 'now' to extrapolate to current datetime, 'last' to extrapolate to the end of the last day and 'none' to disable extrapolation. Interpolation needs to be enabled.") 
        if extrapolate_to == "now" and max_date != None: raise ValueError("Can not use max_date parameter and extrapolate_to='now' at the same time as extrapolation sets max date implicitly.")

        # drop originally empty values
        main_df = pd.DataFrame(user_data).dropna(subset=['coordinate'])

        # convert the timestamp field to a datetime object and a string for get_date_frame
        main_df['datetime'] = pd.to_datetime(main_df['timestamp'], unit='ms').dt.round(f'{time_steps}min')
        # Assuming the first time point is 6:00 AM
        base_time = pd.to_datetime('08:00:00', format='%H:%M:%S').time()
        # Calculate minutes passed since 8 AM
        main_df['minute'] = ((main_df['datetime'] - pd.to_datetime(main_df['datetime'].dt.date.astype(str) + ' ' + base_time.strftime('%H:%M:%S'))).dt.total_seconds() / 60).astype(int) 
        # Map to 15-minute slots
        main_df['minute'] = (main_df['minute'] // 15) * 15
        # drop duplicates
        main_df.drop_duplicates(subset='datetime', keep='first', inplace=True)

        # --- restrict or expand used days and time slots --- 
        # Calculate the earliest and latest dates in the DataFrame
        dataset_earliest_date = main_df['datetime'].min().date()
        dataset_latest_date = main_df['datetime'].max().date()
        used_earlist_date = dataset_earliest_date
        used_latest_date = dataset_latest_date
        # use explicitly given dates instead if possible
        if min_date != None: used_earlist_date = min_date
        if max_date != None: used_latest_date = max_date
        # potentially expand further if used latest date is still before current date
        if extrapolate_to == 'now':
            current_datetime = datetime.datetime.now() 
            if current_datetime.date() > used_latest_date: used_latest_date = current_datetime.date()
        # add extra days (used for shift-based models)
        if extra_days != None:
            used_latest_date = used_latest_date + datetime.timedelta(days=extra_days)
        #print(f"Producing data up to {used_latest_date}.")

        
        # assume time is always explicitly given or implictly set to maximum time range in a day
        used_min_time = min_time
        used_max_time = max_time

        # Create a new datetime range with 15-minute intervals
        time_range = pd.date_range(start=str(used_earlist_date) + " " + str(used_min_time), end=str(used_latest_date) + " " + str(used_max_time), freq=f'{time_steps}T')

        # Create a new DataFrame with the datetime range
        new_df = pd.DataFrame({'datetime': time_range})

        # Merge (outer join) the new DataFrame with the original DataFrame on "timestamp"
        df = pd.merge(main_df, new_df, on='datetime', how='outer')

        # Sort the DataFrame by "timestamp" if needed
        df.sort_values(by='datetime', inplace=True)

        # Reset the index if needed
        df.reset_index(drop=True, inplace=True)

        # drop dates and times outside of desired time frames
        df = df[df['datetime'].dt.date <= used_latest_date]
        df = df[df['datetime'].dt.date >= used_earlist_date]
        df = df[df['datetime'].dt.time <= used_max_time] 
        df = df[df['datetime'].dt.time >= used_min_time]

        # drop unwanted weekdays
        df = df[df['datetime'].dt.weekday.isin(week_days)] 

        # calculate amount of resulting time slots in a day. Used later to transform tensor into 3D.
        n_days = df['datetime'].dt.date.nunique()
        n_slots = int(df.shape[0] / n_days)

        # interpolate/extrapolate coordinates
        extrapolated_slots = 0
        interpolated_percentage = 0
        if interpolation != None:
            interpolation_area = "inside"
            if extrapolate_to == "last":
                interpolation_area = None # doesn't limit pandas interpolation area, effectively including "inside" (interpolation) and "outside" (extrapolation)
            elif extrapolate_to == "now":
                last_data_index = df['coordinate'].last_valid_index()
                latest_data_datetime = df['datetime'][last_data_index]
                current_datetime = datetime.datetime.now()
                if latest_data_datetime < (current_datetime - datetime.timedelta(minutes=time_steps)): # extrapolate if the latest slot has not been filled yet
                    desired_latest_time_slot_index = max(                           
                        df[df["datetime"] < current_datetime].index   
                    )

                    # if there is no data in the last slot, copy last available coordinates before interpolation to change interpolation area
                    desired_slot_element = df.loc[desired_latest_time_slot_index, 'coordinate']
                    desired_slot_is_empty = pd.isna(desired_slot_element)
                    if hasattr(desired_slot_is_empty, '__len__'): desired_slot_is_empty = desired_slot_is_empty.all() # catch inconsistent pandas returns (one bolean if coords=nan, two boleans if coors=[x,y])
                    if desired_slot_is_empty == True:
                        #print(f"Extrapolating from {df.at[last_data_index, 'coordinate']} at {df.at[last_data_index, 'datetime']}")
                        #print(f"Extrapolating to   {df.at[desired_latest_time_slot_index, 'coordinate']} at {df.at[desired_latest_time_slot_index, 'datetime']}")
                        extrapolated_slots = (df.at[desired_latest_time_slot_index, 'datetime'] - df.at[last_data_index, 'datetime']).total_seconds()/(60*15)
                        df.at[desired_latest_time_slot_index, 'coordinate'] = df.at[last_data_index, 'coordinate']
            
            # interpolate (includes extrapolation if activated)
            prev_nan_count = df['coordinate'].isna().sum() if count_interpolation == True else 0
            df['coordinate'].interpolate(interpolation, limit_direction='forward', limit_area=interpolation_area, inplace=True)

            if count_interpolation == True:
                new_nan_count =  df['coordinate'].isna().sum()
                interpolated_slots = prev_nan_count - new_nan_count - extrapolated_slots
                interpolated_percentage = interpolated_slots/len(df['coordinate'])
        
        # find amount of missing trailing slots (needed to interpret results later)
        missing_trailing_slots = 0
        last_filled_slot = df['coordinate'].last_valid_index()
        if last_filled_slot is not None: missing_trailing_slots = df['coordinate'].iloc[df.index.get_loc(last_filled_slot) + 1:].isna().sum()


        # generate lat/long fields
        df[['lat', 'long']] = df['coordinate'].apply(lambda x: pd.Series(x, dtype='float64'))

        date_df = get_date_frame(df)
        #df.drop('datestr', axis=1, inplace=True) # only needed for above func

        ip_df = get_ip_frame(df, home_coordinate, n_clusters=cluster_count, spherical_dist=False)

        # TODO weather - ignores weather for now (needs exact location not only relative map)
        df = pd.concat([df.reset_index(drop=True),
                        ip_df.reset_index(drop=True),
                        # weather_df.reset_index(drop=True),
                        date_df.reset_index(drop=True)], axis=1)

        # normalize distances by transforming them to home distances and scaling them into the 0 - 1 range
        if normalize == True:
            coordinate_min = 0
            coordinate_max = 25000
            coordinate_range = coordinate_max - coordinate_min

            # Normalize time-related features
            df['minute'] = df['minute']/59.0
            df['lat'] = df['lat'].apply(lambda lat: (abs(lat - home_coordinate[0]) / coordinate_range))
            df['long'] = df['long'].apply(lambda long:  (abs(long - home_coordinate[1]) / coordinate_range))
            df['distance_2_home'] = df['distance_2_home'].apply(lambda distance: distance / math.sqrt(coordinate_range**2 + coordinate_range**2))
            home_cutoff = home_cutoff / math.sqrt(coordinate_range**2 + coordinate_range**2)
            df['distance_2_home_standard'] = (df['distance_2_home'] - df['distance_2_home'].mean())/df['distance_2_home'].std(ddof=0)

        if add_temporal_features == True:
            df['momentum'] = df['distance_2_home'].diff()

        df['minute'].fillna(method='ffill', inplace=True)
        df['distance_2_home'].fillna(method='ffill', inplace=True)
        # Accumulate
        x_traj_list = [
            df['minute'].to_numpy()
            ]
        
        if normalize == True:
            x_traj_list.append(df['distance_2_home_standard'].to_numpy())

        if add_temporal_features == True:
            x_traj_list.append(df['momentum'].to_numpy())

        # generate clusters if desired 
        for i in range(cluster_count):
            x_traj_list.append(df['distance_2_ip_' + str(i)].to_numpy())

        # generate tensor for location information from DF
        x_traj = torch.from_numpy(np.stack(x_traj_list).T).float()

        # TODO include this for weather
        # Get full additional information
        # condition_list = df['conditions'].to_list()
        # condition = torch.zeros(len(condition_list), dtype=torch.int64)
        # for i, c in enumerate(set(condition_list)):
        #     condition[np.array(condition_list) == c] = int(i)
        # condition = one_hot(condition)

        is_holiday_np = np.array(df['is_holiday'].to_numpy(), dtype='float')
        is_holiday_values = torch.nn.functional.one_hot(torch.tensor(is_holiday_np[~np.isnan(is_holiday_np)], dtype=torch.int64), 2).float()
        is_holiday = torch.full((is_holiday_np.shape[0], is_holiday_values.shape[1]), torch.nan)
        is_holiday[torch.tensor(~np.isnan(is_holiday_np))] = is_holiday_values
        weak_day_np = np.array(df['weak_day'].to_numpy(), dtype='float')
        weak_day_values = torch.nn.functional.one_hot(torch.tensor(weak_day_np[~np.isnan(weak_day_np)], dtype=torch.int64), 7).float()
        weak_day = torch.full((weak_day_np.shape[0], weak_day_values.shape[1]), torch.nan)
        weak_day[torch.tensor(~np.isnan(weak_day_np))] = weak_day_values
        

        x_additional = torch.concat(
                [
                    is_holiday,
                    weak_day],
                axis=1)
        
        # Label data
        distance_2_home_np = df['distance_2_home'].to_numpy()
        y = torch.from_numpy(np.where(np.isnan(distance_2_home_np), np.nan, distance_2_home_np < home_cutoff)).float()

        if len(x_traj) != (n_days * n_slots) or len(x_additional) != (n_days * n_slots):
            raise RuntimeError("Sanity check for data processing failed.")

        x_input = x_traj.view(n_days, n_slots, -1)
        x_input_additional = x_additional.view(n_days, n_slots, -1)
        target = y.view(n_days, n_slots, -1)
        target = target.contiguous()

        x_input = x_input.nan_to_num(nan=input_replacement_value).contiguous()
        x_input_additional = x_input_additional.nan_to_num(nan=input_replacement_value).contiguous()

        # add additional encoding for time slots
        if add_timeslot_feature == True: 
            timeslot_values = torch.tensor([ts_encoding/(n_slots - 1) for ts_encoding in range(0, n_slots)], device=x_input_additional.device)
            new_feature_tensor = torch.zeros((*x_input_additional.shape[0:-1], 1))
            new_feature_tensor[:, :, 0] = timeslot_values
            x_input_additional = torch.cat((x_input_additional, new_feature_tensor), -1)



        if validation_split_percentage > 0:
            def compute_day_threshold(target, train_val_rat=(1-(validation_split_percentage/100))):
                num_t_slots = (~target.isnan()).sum(axis=1)[:, 0]
                is_containig_val = torch.cumsum(num_t_slots, 0) >= train_val_rat * (~target.isnan()).sum()
                day_threshold = torch.nonzero(is_containig_val).squeeze()[0]
                return day_threshold

            day_threshold=compute_day_threshold(target)
            

            x_train=x_input[:day_threshold]
            x_train_add=x_input_additional[:day_threshold]
            target_train=target[:day_threshold,...,0]
            training_data = TensorDataset(x_train, x_train_add, target_train) 

            x_val=x_input[day_threshold:]
            x_val_add=x_input_additional[day_threshold:]
            target_val=target[day_threshold:,...,0] 
            validation_data = TensorDataset(x_val, x_val_add, target_val) 
            
            data_processing_util.append_dataset_metadata(training_data, extrapolated_slots, interpolated_percentage, missing_trailing_slots, used_min_time, used_max_time, week_days)
            data_processing_util.append_dataset_metadata(validation_data, extrapolated_slots, interpolated_percentage, missing_trailing_slots, used_min_time, used_max_time, week_days)
            
            return [training_data, validation_data]
        else:
            training_data = TensorDataset(x_input, x_input_additional, target[:, ..., 0]) 

            data_processing_util.append_dataset_metadata(training_data, extrapolated_slots, interpolated_percentage, missing_trailing_slots, used_min_time, used_max_time, week_days)

            return [training_data, None]
    
    @staticmethod
    def append_dataset_metadata(dataset, extrapolated_slots, interpolated_percentage, missing_slots, daily_min_time, daily_max_time, weekdays):
        dataset.extrapolated_slots = extrapolated_slots
        dataset.interpolated_percentage = interpolated_percentage
        dataset.missing_slots = missing_slots
        dataset.daily_min_time = daily_min_time
        dataset.daily_max_time = daily_max_time
        dataset.weekdays = weekdays

    @staticmethod
    def get_dataset_metadata(dataset):
        metadata = dict()
        metadata["interpolated_percentage"] = dataset.interpolated_percentage if hasattr(dataset, "interpolated_percentage") else None
        metadata["extrapolated_slots"] = dataset.extrapolated_slots if hasattr(dataset, "extrapolated_slots") else None
        metadata["missing_slots"] = dataset.missing_slots if hasattr(dataset, "missing_slots") else None
        metadata["daily_min_time"] = dataset.daily_min_time if hasattr(dataset, "daily_min_time") else None
        metadata["daily_max_time"] = dataset.daily_max_time if hasattr(dataset, "daily_max_time") else None
        metadata["weekdays"] = dataset.weekdays if hasattr(dataset, "weekdays") else None
        return metadata

    @staticmethod
    def add_padding(data_tuple, padding_length, padding_dim=0, label_tensor_index=-1, label_padding_value=float('nan'), input_padding_value="auto"):
        if input_padding_value == "auto": input_padding_value = data_processing_util.padding_value
        if label_tensor_index != None: label_tensor_index = label_tensor_index if label_tensor_index >= 0 else len(data_tuple) + label_tensor_index

        full_tuple = []
        for index in range(len(data_tuple)):
            # set padding size
            padding_size = list(data_tuple[index].size())
            padding_size[padding_dim] = padding_length

            # generate padding; different values for input tensors and label tensors 
            padding = torch.full(padding_size, input_padding_value, device=data_tuple[index].device) if index != label_tensor_index else torch.full(padding_size, label_padding_value, device=data_tuple[index].device)
            
            # concatenate padding tensor and original tensor
            full_tuple.append(torch.cat((padding, data_tuple[index]), padding_dim))
        return full_tuple

    @staticmethod
    def generate_dataset_metrics(dataset) -> dict:
        dataset_metadata = data_processing_util.get_dataset_metadata(dataset)
        new_metrics = dict()
        
        # output training data stats
        new_metrics["samples"] = len(dataset)
        new_metrics["unlabeled"] = torch.isnan(dataset.tensors[-1])
        new_metrics["labeled_count"] = np.count_nonzero(~new_metrics["unlabeled"])
        new_metrics["unlabeled_count"] = np.count_nonzero(new_metrics["unlabeled"])
        new_metrics["interpolated_percentage"] = dataset_metadata.get("interpolated_percentage", None) if dataset_metadata.get("interpolated_percentage", None) != None else 0
        new_metrics["extrapolated_slots"] = dataset_metadata.get("extrapolated_slots", None) if dataset_metadata.get("extrapolated_slots", None) != None else 0
        new_metrics["total_label_fields_count"] = new_metrics["labeled_count"]  + new_metrics["unlabeled_count"] 
        new_metrics["positive_labeled"] = dataset.tensors[-1] == 1
        new_metrics["positive_count"] = np.count_nonzero(new_metrics["positive_labeled"])

        return new_metrics
    
    @staticmethod
    def log_presence(predictions, start_date, logger, raw_predictions=None, reliability_levels=None, labels=None, interval="15T", start_time="08:00:00", end_time="20:45:00", used_weekdays={0,1,2,3,4,5}):
        data_headers = pd.date_range(start=f"01.01.1999 {start_time}", end=f"01.01.1999 {end_time}", freq=interval).strftime('%H:%M')
        #data_headers = pd.timedelta_range(start=datetime.timedelta(hours=6), end=datetime.timedelta(hours=23, minutes=45), freq='15T').delta.abs().components[0]#strftime('%H:%M')

        has_labels = False
        if labels is not None and len(labels) > 0: 
            if predictions.shape != labels.shape:
                logger.info(f"Model results do not match provided label data. Results can not be verified. Result shape: {predictions.shape} - Label shape: {labels.shape}")
            elif len(labels[~labels.isnan()]) == 0:
                logger.info("No Label Data has been provided, so results can not be verified.")
            else:
                has_labels = True
                logger.info("Fitting label data has been provided.")

        total_predictions = 0
        nan_labels = 0
        guessed_home = 0
        was_home = 0
        overlap_home = 0
        false_positives = 0
        true_positives = 0
        index = 0
        last_date = start_date - datetime.timedelta(days=1) #end_date - datetime.timedelta(days=output_days_count)
        for result in predictions:
            index += 1
            last_date = last_date + datetime.timedelta(days=1)
            while last_date.weekday() not in used_weekdays and index != 1:
                last_date = last_date + datetime.timedelta(days=1)

            logger.headline(f"Result {index}, date {last_date}:")
            # try to print in formatted style if headers are available
            if (data_headers is not None) and (len(data_headers) == len(result)):
                for result_index, data_header in enumerate(data_headers):
                    i_result = result[result_index].item()
                    i_raw_result = raw_predictions[index - 1][result_index].item()
                    i_result_string_num = str(round(i_result, 3))
                    i_raw_result_string_num = str(round(i_raw_result, 3))
                    i_result_string_eval = "Unkown "
                    if i_result == 1 or i_result == True:
                        i_result_string_eval = "Present"
                        guessed_home += 1
                    elif i_result == 0 or i_result == False:
                        i_result_string_eval = "Absent "
                        guessed_home += 1

                    log_string = f"{data_header}: "
                    if raw_predictions is not None: log_string += f"{i_raw_result_string_num:^6} => "
                    log_string += f"{i_result_string_num:^6} => "
                    log_string += f"{i_result_string_eval}"

                    if has_labels: 
                        i_label = labels[index-1][result_index].item()
                        i_label_string_num = str(i_label)
                        i_label_string_eval = "-------"
                        if i_label == 1:
                            i_label_string_eval = "Present"
                            was_home += 1  
                        elif i_label == 0:
                            i_label_string_eval = "Absent "

                        i_label_match_string = "Incorrect" 
                        
                        if i_label != 0 and i_label != 1:
                            i_label_match_string = "---------"
                            nan_labels += 1
                        if i_label_string_eval != "-------":
                            # correct prediction
                            if i_label_string_eval == i_result_string_eval:
                                i_label_match_string = "Correct"
                                overlap_home += 1

                                if i_result_string_eval == "Present":
                                    true_positives += 1
                            # false positives
                            elif i_result_string_eval == "Present":
                                false_positives += 1


                        log_string += f"  <>  {i_label_string_eval} = {i_label_string_num} = Label"
                        log_string += f"  <>  {i_label_match_string}"

                    if reliability_levels is not None: log_string += f" | Reliability level: {reliability_levels[index -1][result_index]}"

                    logger.info(log_string)
                    total_predictions += 1
            # otherwise just print output tensors
            else:    
                logger.info(str(result))

        if has_labels:
            sk_mask = ~labels.isnan()
            filtered_results = (predictions[sk_mask] > 0.5).type(torch.IntTensor).detach().cpu().numpy()
            filtered_labels = labels[sk_mask].detach().cpu().numpy()
            sk_precision = round(100 * precision_score(filtered_labels, filtered_results), 3)
            sk_accuracy = round(100 * accuracy_score(filtered_labels, filtered_results), 3)

            precision = round(100 * true_positives/(false_positives + true_positives), 3) if (false_positives + true_positives) > 0 else 0
            accuracy = round(100 * overlap_home/(total_predictions - nan_labels), 3) if (total_predictions - nan_labels > 0) else 0

            differences = (predictions - labels)
            average_difference = torch.mean(torch.abs(differences.nan_to_num()))

            logger.headline("Overall result:")
            logger.info(f"Average difference before classification:    {average_difference}")
            logger.info(f"Model predicted that user would be home      {guessed_home} times")
            logger.info(f"User was actually home                       {was_home} times")
            logger.info(f"Correct positive predictions:                {true_positives:<5} |  {precision}% precision")
            logger.info(f"Correct positive and negative predictions:   {overlap_home:<5} |  {accuracy}% accuracy")

            if sk_precision != precision or sk_accuracy != accuracy:
                logger.critical(f"Warning: Metrics do not match. SK-Accuracy: {sk_accuracy}, SK-Precision: {sk_precision}")


    @staticmethod
    def interpret_shift_string(shift_string):
        result = None
        if type(shift_string) is str:
            if str.count(shift_string, ":") > 0:
                shift_string = str.split(shift_string, ":")
                result = (int(shift_string[0]), int(shift_string[1]))
            else:
                result = (int(shift_string), 0)
        elif type(shift_string) is tuple:
            result = shift_string
        elif type(shift_string) is int:
            result = (shift_string, 0)
        elif shift_string == None:
            result = None
        else:
            raise ValueError("Invalid shift string has been provided.")

        if result != None and result[1] != 0: 
            raise ValueError("Shifting partial days is not fully supported.")
        else:
            return result
        

