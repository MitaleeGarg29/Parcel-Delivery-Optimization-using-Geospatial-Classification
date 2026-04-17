import json
import datetime

from torch.utils.data import TensorDataset

from .internal.data_procurer import data_procurer
from .internal.data_processing import data_processing_util
from .internal.artificial_data import artificial_data_util


class artificial_singleuser(data_procurer):
    """
    A data procurer subclass meant for analytic trainings and evaluations on artificial data.
    Deterministically generates data based on the settings given in the class variables.

    Will reliably reproduce the same patterns for the same user ID.
    Each user will have an arbitrary amount of weekly absences and presences. 
    Each user as an arbitrary home address. 
    Each absence will lead the user to an arbitrarily generated external location.
    Each presence will last for an arbitrary amount of timeslots. 
    Between absences and presences, a user coordinates are interpolated linearily, simulating the users movement between locations.

    Uses the standard user data processing to generate input tensors containing coordinates, distances, weekdays and holidays, as well as label/ground truth tensors defining presence or absence. 
    
    Loads and processes a users training and validation data, given in the form of TensorDatasets.
    """

    start_date = "2023-12-01"
    start_time = "08:00:00"
    end_date = "2024-03-23"
    end_time = "20:45:00"
    timeslots_per_day = 52
    used_weekdays = {0,1,2,3,4,5}
    max_regular_absences = 18           # Maximum of weekly external locations a user may visit. Exact number is determined at random for every user.
    max_home_timeslots = 14             # Maximum of how long the user will stay at home after each regular absence.
    irregular_absences = False          # If True, irregular absences are generated in addition to regular ones. Up to one per month, for between 3 and 14 days at a time, decided at random, somewhere within each 30 day window. Untested!

    cut_last_day_to = 28                # amount of timeslots the last day should be reduced to in order to simulate partial days for a following prediction

    generate_readable_fields = True     # Generates additional fields that for manual analysis in CSV files

    def _load(self, **kwargs) -> str:
        # sanity check params
        if artificial_data_util.sanity_check_params(self.used_weekdays, self.timeslots_per_day, self.max_regular_absences, self.max_home_timeslots) == False:
            raise ValueError(f"Invalid parameters. User may not be able to be away that often while still being at home for {self.max_home_timeslots} time slots every time.")

        artifical_user_id = kwargs['user']

        # log used settings
        self.logger.headline("Generating artifical users.")
        self.logger.info("Settings:")
        self.logger.info(f" - User ID: {artifical_user_id}")
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

        # generate data
        i_user_home, i_user_data, i_day_count, i_week_count, i_external_locations_count, i_irregular_absences_count, i_average_presence_duration = artificial_data_util.generate_user(artifical_user_id, self.start_date, self.start_time, self.end_date, self.end_time, self.timeslots_per_day, self.used_weekdays, self.max_regular_absences, self.max_home_timeslots, self.irregular_absences, self.generate_readable_fields)

        if self.cut_last_day_to != None:
            last_timeslots = self.timeslots_per_day - self.cut_last_day_to
            if last_timeslots > 0:
                i_user_data.iloc[-last_timeslots:] = float('nan')
            self.logger.info(f" - Cutting last day from slot: {last_timeslots}")

        # save user
        user_homes.append(i_user_home)
        user_dataframes.append(i_user_data)

        # log user stats
        self.logger.info(f"Stats for artifical user {artifical_user_id}:", True)
        self.logger.info(f" - Home address: {user_homes[0]}:")
        self.logger.info(f" - Generated timespan: {i_day_count} days/{i_week_count} weeks:")
        self.logger.info(f" - Weekly absences: {i_external_locations_count}:")
        self.logger.info(f" - Average continous presence: {i_average_presence_duration} time slots")
        self.logger.info(f" - Irregular absences: {i_irregular_absences_count}:")

        
        # transform to string to imitate real raw format
        raw_data = artificial_data_util.dataframes_to_string(user_homes, user_dataframes, 1)

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
