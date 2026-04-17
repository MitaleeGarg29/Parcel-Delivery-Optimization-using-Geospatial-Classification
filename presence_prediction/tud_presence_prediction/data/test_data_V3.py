import torch
import random

from torch.utils.data import TensorDataset

from .internal.data_procurer import data_procurer as dp


class test_data_V3(dp):
    """
    A data procurer subclass meant for trivial training analysis. 

    Simply generates tensors filled with numbers indiciating the position of each timeslot and a user ID.
    Can be useful the observe and test variation generation and batch generation performed by the training preperation on data retrieved by data procurers.
    When logging the contents of each batch, e.g. by using the test logging functions within the model_util.py file, one can ensure that the expected variations and batches are created by inspecting the timeslot index and user index for all sequences in each training step.
    """

    days_per_user_min = 30
    days_per_user_max = 90
    timeslots_per_day = 2
    user_count = 4
    generate_validation_data = False

    def _load(self, **kwargs) -> str:
        return "Test"

    
    def _process(self, loaded_raw_data, **kwargs) -> list[tuple[TensorDataset]]:
        
        output = []

        for user_index in range(1, self.user_count + 1):
            random.seed(user_index)
            training_user_day_count = random.randint(self.days_per_user_min, self.days_per_user_max)

            training_input = torch.arange(start=0, end=training_user_day_count*self.timeslots_per_day, step=0.5).float().reshape(training_user_day_count, self.timeslots_per_day, 2)
            training_input[:, :, 1] = user_index
            training_label = torch.randint(low=0, high=2, size=(training_user_day_count, self.timeslots_per_day)).float()

            training_data = TensorDataset(training_input, training_input.clone(), training_label)

            validation_data = None
            if self.generate_validation_data == True:
                random.seed(user_index + 0.5)
                validation_user_day_count = random.randint(self.days_per_user_min, self.days_per_user_max)

                validation_input = torch.arange(start=0, end=validation_user_day_count*self.timeslots_per_day, step=0.5).float().reshape(validation_user_day_count, self.timeslots_per_day, 2)
                validation_input[:, :, 1] = user_index
                validation_label = torch.randint(low=0, high=2, size=(validation_user_day_count, self.timeslots_per_day)).float()

                validation_data = TensorDataset(validation_input, validation_input.clone(), validation_label)

            output.append([training_data, validation_data])

            print(f"User {user_index} training input shape: {training_input.shape}")
            #print(f"User {user_index} training input data: {training_input}")
            print(f"User {user_index} training label shape: {training_label.shape}")
            #print(f"User {user_index} training input data: {validation_label}")

        return output
    
