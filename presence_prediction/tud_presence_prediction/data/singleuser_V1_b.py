import torch
import numpy as np
import pandas as pd
import requests
import json

from torch.utils.data import TensorDataset

from .internal.data_procurer import data_procurer as dp
from .internal.crawler.date.load_holiday_data import get_date_frame
from .internal.crawler.interest_point.load_ip_data import get_ip_frame

class singleuser_V1_b(dp):
    usr_id = 'CUSTOMER_78db5215-de12-460d-97e7-23000d9b4950' #'4147127f-1c47-4de8-8e43-58f19d41b418'
    geo_hash_home = '2zudz'

    def _load(self, **kwargs) -> list[tuple[TensorDataset]]:
        
        urls = ['https://rl.green-convenience.com/api/v1/positions/convert/' + self.geo_hash_home,
                "https://rl.green-convenience.com/api/v1/positions/preprocess/" + self.usr_id + "/" + self.geo_hash_home]

        # Get data from database
        #json_data = []
        text_data = "["
        for index, url in enumerate(urls):
            response = requests.get(url)

            # Check if the request was successful
            if response.status_code == 200:
                #json_data.append(response.json())
                #test_json = response.json()
                text_data += response.text + ("," if index < (len(urls) -1) else "")
            else:
                raise RuntimeError(f"Failed to retrieve data from the URL. Status code: {response.status_code}")
                return None

        text_data += "]"    

        return text_data

    
    def _process(self, loaded_raw_data, **kwargs) -> list[tuple[TensorDataset]]:
        json_data = json.loads(loaded_raw_data)
        
        home_coordinate = json_data[0]['coordinate']
        # TODO: Filter raw data further. Seems to rarely contain some datetimes twice (e.g. line 6881, 6882)
        main_df = pd.DataFrame(json_data[1]).dropna(subset=['coordinate'])

        # Step 3: Convert the timestamp field to a datetime object
        main_df['datetime'] = pd.to_datetime(main_df['timestamp'], unit='ms').dt.round('15min')
        main_df.drop_duplicates(subset='datetime', keep='first', inplace=True)
        main_df['datestr'] = main_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

        main_df[['lat', 'long']] = main_df['coordinate'].apply(lambda x: pd.Series(x, dtype='float64'))

        date_df = get_date_frame(main_df)

        # TODO Define kmeans hyper param
        n_clusters = 0
        # TODO: check if distance is correct. Distances are currently significantly higher than differences in lat/long
        ip_df = get_ip_frame(main_df, home_coordinate, n_clusters=n_clusters, spherical_dist=False)

        # TODO weather - ignores weather for now (needs exact location not only relative map)
        df = pd.concat([main_df.reset_index(drop=True),
                        ip_df.reset_index(drop=True),
                        # weather_df.reset_index(drop=True),
                        date_df.reset_index(drop=True)], axis=1)

        # TODO: Old TODOS:
        # - "zero padding of data"?
        # - Ensure "timestamp" column is in datetime format
        # df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Calculate the earliest and latest dates in the DataFrame
        earliest_date = df['datetime'].min().date()
        latest_date = df['datetime'].max().date()

        # Create a new datetime range with 15-minute intervals from 6 am to 12 pm
        # TODO: Make sure this acually encompasses the last 15 min or remove them completely?
        # TODO: Potentially restrict time frame to the ones actual data is present for? Currently 8:00 to 20:45
        time_range = pd.date_range(start=str(earliest_date), end=str(latest_date) + ' 23:59:00', freq='15T')

        # Create a new DataFrame with the datetime range
        new_df = pd.DataFrame({'datetime': time_range})

        # Merge (outer join) the new DataFrame with the original DataFrame on "timestamp"
        df = pd.merge(df, new_df, on='datetime', how='outer')

        # Sort the DataFrame by "timestamp" if needed
        df.sort_values(by='datetime', inplace=True)

        # Reset the index if needed
        df.reset_index(drop=True, inplace=True)

        # Only safe df for times between 6am and 12pm
        df = df[(df['datetime'].dt.hour >= 6)]
        n_days = df['datetime'].dt.date.nunique()
        n_slots = int(df.shape[0] / n_days)

        # Get full trajectory information
        x_traj_list = [
            df['lat'].to_numpy(),
            df['long'].to_numpy(),
            df['distance_2_home'].to_numpy()]

        # TODO Old: Define kmeans hyper param
        # TODO PB: Also include hotspots locations for multiple users so information of one user can be used to infer information about new users if they visit the same location?
        for i in range(n_clusters):
            x_traj_list.append(df['distance_2_ip_' + str(i)].to_numpy())
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
                    # torch.from_numpy(np.stack([df['temp'].to_numpy(), df['cloudcover'].to_numpy()]).T),
                    # condition,
                    is_holiday,
                    weak_day],
                axis=1)
        
        # Label data
        distance_2_home_np = df['distance_2_home'].to_numpy()
        y = torch.from_numpy(np.where(np.isnan(distance_2_home_np), np.nan, distance_2_home_np < 500)).float()

        x_input = x_traj.view(n_days, n_slots, -1)
        x_input_additional = x_additional.view(n_days, n_slots, -1)
        target = y.view(n_days, n_slots, -1)
        target = target.contiguous()

        # # Step 4: Create a new column 'datestr' as a string representation of the datetime
        # df['datestr'] = df['date_time']

    
        x_input = x_input.nan_to_num().contiguous()
        x_input_additional = x_input_additional.nan_to_num().contiguous()

        TRAIN_PERCENT=80
        def compute_day_threshold(target, train_val_rat=0.8):
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
        target_val=target[day_threshold:,...,0] # TODO: Why this syntax?
        validation_data = TensorDataset(x_val, x_val_add, target_val) 
        
        return [training_data, validation_data]
    
    def _set_data_headers(self):
        data_headers = [
            #["Lat", "Long", "From Home", "From POI 1", "From POI 2", "From POI 3", "From POI 4", "From POI 5", "From POI 6", "From POI 7", "From POI 8", "FromPOI 9", "From POI 10"],
            ["Lat", "Long", "From Home"],
            ["Not Holiday", "Is Holiday 2", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            ["Label"]
        ]

        return data_headers
