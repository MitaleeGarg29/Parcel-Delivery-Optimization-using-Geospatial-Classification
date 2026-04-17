import torch
import numpy as np
import pandas as pd
import requests
import json
import datetime
import holidays

from torch.utils.data import TensorDataset

from .internal.data_procurer import data_procurer as dp
from .internal.crawler.date.load_holiday_data import get_date_frame
from .internal.crawler.interest_point.load_ip_data import get_ip_frame

class prediction_test_empty(dp):
    time_slots = 72
    position_data_slots = 13
    time_data_slots = 9

    def _load(self, **kwargs) -> str:
        text_data = ""

        # TODO: load data in string format
        date_object = datetime.date.fromisoformat(kwargs["date"])
        self.date = date_object

        return date_object.isoformat() + ";" + str(kwargs["days"])

    
    def _process(self, loaded_raw_data, **kwargs) -> list[tuple[TensorDataset]]:
        prediction_date = datetime.date.fromisoformat(loaded_raw_data.split(";")[0])
        prediction_duration = int(loaded_raw_data.split(";")[1])
        
        position_data = torch.full((prediction_duration, self.time_slots, self.position_data_slots), float(0))
        time_data = torch.full((prediction_duration, self.time_slots, self.time_data_slots), float(0)) 
        label_data = torch.full((prediction_duration, self.time_slots, 1), float("nan")) 
        #label_data = torch.full((0), float(0))
        #label_data = torch.from_numpy(np.empty(0))
        #label_data = torch.from_numpy(np.fromiter([]))

        ger_holidays = holidays.Germany()
        for day in range(prediction_duration):
            i_prediction_date = prediction_date + datetime.timedelta(days=day)

            is_holiday = i_prediction_date.strftime('%Y-%m-%d %H:%M:%S') in ger_holidays
            is_holiday_tensor = torch.nn.functional.one_hot(torch.tensor(is_holiday), 2).float()
            weekday = i_prediction_date.weekday()
            weekday_tensor = torch.nn.functional.one_hot(torch.tensor(weekday), 7).float()

            time_data[day][0:72] = torch.concat([is_holiday_tensor, weekday_tensor])
            #time_data[day][0][1:-1] = weekday_vector

        prediction_input = TensorDataset(position_data, time_data, label_data) 
        
        return [prediction_input, None]
    
    def _set_data_headers(self):
        data_headers = [
            ["Lat", "Long", "From Home", "From POI 1", "From POI 2", "From POI 3", "From POI 4", "From POI 5", "From POI 6", "From POI 7", "From POI 8", "From POI 9", "From POI 10"],
            ["Is Holiday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            ["Label"]
        ]

        return data_headers

