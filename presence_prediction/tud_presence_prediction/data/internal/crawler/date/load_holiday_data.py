import pandas as pd
import holidays
from datetime import datetime


def get_date_frame(df, country='Germany'):
    """
    Loads corresponding date information from dataframe
    :param df: data frame containing a 'datestr' field
    :return: panda data frame with week-day and holiday information
    """
    #filtered_df = df[df['datestr'] != float('nan')]
    #filtered_df = df.dropna(subset=['datestr'])
    date_key = "datestr"
    if "datestr" not in df.columns and "datetime" in df.columns:
        date_key = "datetime"


    ger_holidays = getattr(holidays, country)()
    is_holiday = [date in ger_holidays for date in df[date_key].dt.strftime('%Y-%m-%d %H:%M:%S').to_list()]
    weak_day = [datetime.fromisoformat(date).weekday() for date in df[date_key].dt.strftime('%Y-%m-%d %H:%M:%S').to_list()]
    date_df = pd.DataFrame(data=dict(weak_day=weak_day, is_holiday=is_holiday))

    return date_df