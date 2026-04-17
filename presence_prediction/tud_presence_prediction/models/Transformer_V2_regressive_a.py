import torch
import pytorch_lightning as pl
import torch.nn as nn
import math

from .internal.polar import PolarCoordinateEmbedding
from .internal.Time2Vec import Time2Vec
from .internal.model_util import model_util
from tud_presence_prediction.helpers.profiling.time_profiling import TimeProfiler


class Transformer_V2_regressive(pl.LightningModule):
    """
    A transformer-based model capable of performing regressive predictions of variable length. 
    It effectively attempts to first continue a given time series and to then generate values indicating the likelyhood of presences within the returned timeslots.
    
    During training time, this model separates the last day(s) of each input sequence and attempts to predict them. 
    Its parameters determine the used regressive method(s), prediction step size, total prediction length and masking. All of these may have a strong impact on training results.

    During prediction time, functions provided by the 'model_util.py' file may be used to set this models desired prediction length. Other parameters will be loaded from the checkpoint generated during training.
    """

    def __init__(self, future_unknown_feature_count, future_known_covariate_feature_count, input_dim_3, hidden_dim=10, classification_layers=1, positional_encoding="time2vec", add_day_feature=False, transformer_head_count=4, transformer_linear_dim=2048, transformer_layer_count=6, transformer_activiation_func="relu", transformer_layer_norm_first=False, transformer_dropout=0.1, use_combined_loss=False, secondary_loss_func=torch.nn.MSELoss, secondary_loss_weight=0.5, training_mode="auto-regressive", change_to_auto_regression=-1, prediction_step_length=(0,26), prediction_total_length=1, use_mask="custom step", pos_weight=1, empty_slot_value=-1, use_optimizer="AdamW", optimizer_beta1=None, optimizer_beta2=None, weight_decay=None, fixed_learning_rate=0.001, log_LR=True, use_LR_scheduler=True, LR_scheduler_min=0.000000175, LR_scheduler_max=0.00009, LR_scheduler_convergence=0.000016, LR_increase_phase_percentage=0.05):
        """
        Initializes the model with the given parameters.

        Args:
            training_mode (str):
                Defines the mode used for multiple internal predictions to accumelate a complete prediction of a desired length. Available values:
                    "auto-regressive":  Subsequent predictions are performed based on the models last outputs. Also known as "free-running". This mode is always used for productive predictions.
                    "regressive":       Subsequent predictions are performed based on given ground-truth values. Also known as "teacher forcing". This may speed up the training initially. This mode can only be used for training.
            change_to_auto_regression (float or int):
                If set to -1, only the given training mode is used. If set to a value between 0 and 1, the training loop will switch to the auto-regressive mode after the given percentage of epochs has passed, relative to the given total amount of epochs within the Lightning trainer.
            prediction_step_length (tuple[int]):
                Defines the internally used prediction length for one prediction, given as (days, timeslots). Multiple of these predictions will be performed to generate one full prediction of the given total length, if neccessary.         
            use_mask (str):
                Defines the mask to be used by the transformer. Available values:
                    None:           No mask is used
                    "subsequent":   Torch's default subsequent mask is used. The masking is generated in steps of one timeslot. Each timeslot may only attent to any previous timeslot. 
                    "custom step":  A custom masking. It is generated dynamically to fit the "prediction_step_length" parameter. The amount of masking is reduced. Each timeslot is allowed to attend to any previous timeslot and all timeslots within one step size.
        """
        
        super(Transformer_V2_regressive, self).__init__()
        extra_dim = 0
        if add_day_feature == True: extra_dim = 1

        transformer_input_dims = hidden_dim*2
        if positional_encoding == "time2vec":
            self.time2vec = Time2Vec(future_unknown_feature_count, hidden_dim)
            self.covariates_embedding = nn.Linear(future_known_covariate_feature_count + extra_dim, hidden_dim*2)
        elif positional_encoding == "polar":
            self.polar_embedding = PolarCoordinateEmbedding(hidden_dim)
            transformer_input_dims = hidden_dim * 2
        elif positional_encoding == "linear":
            self.input_encoding1 = nn.Linear(future_known_covariate_feature_count + future_unknown_feature_count + extra_dim, hidden_dim * 2)
            self.input_encoding2 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        elif positional_encoding == None:
            transformer_input_dims = future_known_covariate_feature_count + future_unknown_feature_count + extra_dim
            transformer_head_count = 4 # TODO: more dynamic approach
        else:
            raise ValueError("Input encoding not recognized.")
        
        
        self.transformer = nn.Transformer(transformer_input_dims, nhead=transformer_head_count, dim_feedforward=transformer_linear_dim, activation=transformer_activiation_func, num_decoder_layers=transformer_layer_count, num_encoder_layers=transformer_layer_count, norm_first=transformer_layer_norm_first, dropout=transformer_dropout, batch_first=True)

        self.pos_weight = torch.tensor([pos_weight]).cuda() if torch.cuda.is_available() else torch.tensor([pos_weight])
        if use_combined_loss == True: self.transformer_loss_fun = secondary_loss_func() #nn.L1Loss()
        self.loss_fun = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        if classification_layers > 1:
            layer_list = []
            for layer_index in range(0, classification_layers - 1):
                layer_list.append(torch.nn.Linear(transformer_input_dims, transformer_input_dims))
            self.linear_classification = torch.nn.Sequential(*layer_list)
        self.fc = nn.Linear(transformer_input_dims, 1)

        self.classification_layers = classification_layers
        self.add_day_feature = add_day_feature
        self.secondary_loss_weight = secondary_loss_weight
        self.use_combined_loss = use_combined_loss
        self.training_mode = training_mode                          # supports "regressive" and "auto-regressive"
        self.change_to_auto_regression = change_to_auto_regression  # change to auto regression after percentage of the training time has passed
        self.prediction_step_length = prediction_step_length
        self.prediction_total_length = prediction_total_length
        self.use_mask = use_mask
        self.empty_slot_value = empty_slot_value
        self.positional_encoding = positional_encoding
    
        self.use_optimizer = use_optimizer
        self.optimizer_beta1 = optimizer_beta1
        self.optimizer_beta2 = optimizer_beta2
        self.weight_decay = weight_decay
        self.fixed_learning_rate = fixed_learning_rate
        self.log_LR = log_LR
        self.use_LR_scheduler = use_LR_scheduler
        self.LR_scheduler_min = LR_scheduler_min
        self.LR_scheduler_max = LR_scheduler_max
        self.LR_scheduler_convergence = LR_scheduler_convergence
        self.LR_increase_phase_percentage = LR_increase_phase_percentage

        self.max_input_length = None 

        self.hparams["data_cut"] = self.prediction_total_length

        model_util.configure_logging(self, log_lr = self.log_LR, additional_metrics=(None if use_combined_loss == False else ["TimeSeriesLoss",]))
        self.save_hyperparameters()

    def forward(self, x, x_add=None, batch_idx=None, mode="auto-regressive", gold_token_x=None, gold_token_x_add=None, remove_empty_trailing_slots=True, roll_output=True, adjust_input_length=False):
        #print(f"Calling forward method with mode: {mode}")
        if x_add == None: 
            x, x_add, label = x

        # TODO:
            # - Masking of padding; currently there are no padded users and a test-case must be created first. Then, appropriate masking for leading values of -1 needs to be created. The PyTorch transformer already offers masking capabilities that can then be used.

        # remove empty slots at the end of the input to make sure the prediction begins at the earliest given time (needs to be determined before positional encoding)
        slots_to_remove = 0
        if remove_empty_trailing_slots == True:
            if len(x.shape) > 3: 
                raise ValueError("Removal of empty trailing slots is only supported for a batch size of one and is intended to be used for predictions only.")
            else:
                slots_to_remove = self.find_trailing_empty_slots(x)
        elif roll_output == True:
            raise ValueError("Output may only be adjusted if empty slots are removed.")

        # chosen positional encoding
        x = self.encode_position(x, x_add)

        # prepare gold tokens
        if mode == "regressive":
            if gold_token_x == None or gold_token_x_add == None: 
                raise ValueError("Can not perform regressive mode predictions without gold tokens being provided.")
            else:
                gold_token_x = self.encode_position(gold_token_x, gold_token_x_add)
                gold_token_x = gold_token_x.flatten(-3, -2)      

        # flatten to enable masking
        prev_size = x.size()    # [batch, day, time, feature] or [day, time, feature] 
        prev_timeslot_count = prev_size[-2]
        prediction_horizon = self.prediction_total_length * prev_timeslot_count                                        # "intended days" x "hours in a day"
        prediction_step = self.prediction_step_length[0] * prev_timeslot_count + self.prediction_step_length[1]        # "intended days" x "hours in a day" + "intended hours"
        x = x.flatten(-3, -2)   # [batch, day x time, feature] or [day x time, feature]    

        #print(f"prev_size after: {prev_size}")
        #print(f"prev_timeslot_count after: {prev_timeslot_count}")
        if slots_to_remove != 0:
            x = x[..., :-slots_to_remove, :]
            if gold_token_x_add != None: gold_token_x = gold_token_x[..., :-slots_to_remove, :]

        # Print prediction_horizon and prediction_step
        #print(f"prediction_horizon: {prediction_horizon}")
        #print(f"prediction_step: {prediction_step}")
        # generate next values regressively
        transformer_prediction = []
        if mode == "regressive" or mode == "auto-regressive":
            # past values; starts with given values and will be expanded progressively
            past = x[..., :-prediction_step:, :]
            # last step output will be used for decoder input
            last_step_output = x[..., -prediction_step:, :] # setting this once makes the transformer always use a decoder length of prediction_step

            step_counter = 0
            for index in range(math.ceil(prediction_horizon / prediction_step)):
                
                # Debugging: Print shapes before passing to transformer
                #print(f"Step {index}:")
                #print(f"past shape: {past.shape}")
                #print(f"last_step_output shape: {last_step_output.shape}")
                # generate source mask and predict with internal transformer
                step_output = None
                torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
                if self.use_mask == None:
                    step_output = self.transformer(past, last_step_output)
                elif self.use_mask == "custom step": 
                    src_mask = self.generate_custom_causal_mask(past.shape[-2], prediction_step, self.device)
                    if torch_version >= (2, 0):
                        step_output = self.transformer(past, last_step_output, src_mask=src_mask, src_is_causal=True)
                    else:
                        step_output = self.transformer(past, last_step_output, src_mask=src_mask)
                elif self.use_mask == "subsequent":
                    src_mask = torch.nn.Transformer.generate_square_subsequent_mask(past.shape[-2], self.device)
                    if torch_version >= (2, 0):
                        step_output = self.transformer(past, last_step_output, src_mask=src_mask, src_is_causal=True)
                    else:
                        step_output = self.transformer(past, last_step_output, src_mask=src_mask)
                else:
                    raise ValueError("Invalid setting for input masking.")

                # update past to include last step
                if mode == "auto-regressive": 
                    past = torch.cat((past, last_step_output), -2)
                    last_step_output = step_output
                elif mode == "regressive": 
                    past = torch.cat((past, last_step_output), -2)
                    last_step_output = gold_token_x[..., index*prediction_step:((index + 1) * prediction_step), :]

                # Debugging: Print shapes after update
                #print(f"Updated past shape: {past.shape}")
                #print(f"Updated last_step_output shape: {last_step_output.shape}")
                # cut size to match original input size
                if adjust_input_length == True and self.max_input_length != None and past.size(-2) > self.max_input_length * prev_timeslot_count:
                    past = past[..., last_step_output.size(-2):, :]
                    
                # remember all results for output
                transformer_prediction.append(step_output)
                step_counter += 1
        else:
            raise ValueError("Mode string was not recognized.")
        
        # concat all predictions 
        transformer_prediction = torch.cat(transformer_prediction, -2)

        # drop predictions that were created extra because the step size was not divisible by the prediction horizon
        transformer_prediction = transformer_prediction[..., :prediction_horizon, :]
        self.last_transformer_prediction = transformer_prediction

        # final step(s); may be expanded in the future
        classification_results = transformer_prediction
        if self.classification_layers > 1:
            classification_results = self.linear_classification(classification_results)
        output = self.fc(classification_results) 

        # move prediction forward to account for partial days that may have been present before the prediction started (e.g. if 1/3 of a day has been provided as input, fill the remaining 2/3 with the first generated values and set the first 1/3 to nan as they don't represent a prediction)
        if remove_empty_trailing_slots == True and roll_output == True:
            given_partial_day_slots = prev_timeslot_count - slots_to_remove
            output = output.roll(given_partial_day_slots, dims=-2)
            #output[:given_partial_day_slots, :] = float('nan')
        #print(f"output shape: {output.shape}")
        # restore original size
        new_size = list(prev_size)      # timeslots and batches should be restored to original values 
        new_size[-3] = -1               # day count to be set automatically,
        new_size[-1] = 1                # feature dim changed to 1 after the final prediction step 
        output = output.view(new_size)  
        #print(f"Final output shape: {output.shape}")
        return output

    def training_step(self, batch, batch_idx):
        #print(f"Starting training_step with batch_idx: {batch_idx}")
        x, x_add, target = batch

        self.max_input_length = x.size(-3) - self.prediction_total_length
        pred = self(
            x[..., :-self.prediction_total_length, :, :], 
            x_add[..., :-self.prediction_total_length, :, :], 
            batch_idx, 
            mode = self.training_mode, 
            gold_token_x = x[..., -self.prediction_total_length:, :, :], 
            gold_token_x_add = x_add[..., -self.prediction_total_length:, :, :],
            remove_empty_trailing_slots=False,
            roll_output=False
        )

        target = target[..., -self.prediction_total_length:, :]
        mask = ~target.isnan()

        filtered_pred = pred.squeeze(len(pred.shape)-1)[mask]
        filtered_target = target[mask]

        # calculate additional loss for transformer time series prediction
        additional_metrics = None
        transformer_loss = None
        if self.use_combined_loss == True:
            encoded_input_x_full = self.encode_position(x, x_add)
            transformer_target = encoded_input_x_full[..., -self.prediction_total_length:, :, :].flatten(-3, -2)
            transformer_loss = self.transformer_loss_fun(self.last_transformer_prediction, transformer_target)
            additional_metrics = dict()
            additional_metrics["TimeSeriesLoss"] = transformer_loss.item()

        loss = self.loss_fun(filtered_pred, filtered_target)

        binary_pred = (torch.sigmoid(pred) > 0.5).int() #.type(torch.IntTensor)
        filtered_binary_pred = binary_pred[mask]

        
        model_util.log(self, loss, filtered_binary_pred, filtered_target, "train", additional_metrics=additional_metrics)
        model_util.batch_step(self, self.log_LR, self.use_LR_scheduler)

        if self.use_combined_loss == True: loss += transformer_loss*self.secondary_loss_weight

        if self.change_to_auto_regression != -1:
            if self.training_mode == "regressive":
                if self.trainer.current_epoch/self.trainer.max_epochs >= self.change_to_auto_regression:
                    self.training_mode = "auto-regressive"

        return loss


    def validation_step(self, batch, batch_idx):
        print(f"Starting validation_step with batch_idx: {batch_idx}")
        x, x_add, target = batch

        self.max_input_length = x.size(-3) - self.prediction_total_length
        pred = self(
            x[..., :-self.prediction_total_length, :, :], 
            x_add[..., :-self.prediction_total_length, :, :], 
            batch_idx, 
            mode = "auto-regressive", 
            gold_token_x = x[..., -self.prediction_total_length:, :, :], 
            gold_token_x_add = x_add[..., -self.prediction_total_length:, :, :],
            remove_empty_trailing_slots=False,
            roll_output=False
        )

        target = target[..., -self.prediction_total_length:, :]
        mask = ~target.isnan()

        filtered_pred = pred.squeeze(len(pred.shape)-1)[mask]
        filtered_target = target[mask]

        loss = self.loss_fun(filtered_pred, filtered_target)

        binary_pred = (torch.sigmoid(pred) > 0.5).int() #.type(torch.IntTensor)
        filtered_binary_pred = binary_pred[mask]

        model_util.log(self, loss, filtered_binary_pred, filtered_target, "val")

        return {'val_loss': loss}

    # creates a mask like torch.triu, but allows steps larger than 1
    def generate_custom_causal_mask(self, size, step_size, device):
        mask = torch.full((size, size), float('-inf'), device=device)
        for index in range(1, size):
            rows = mask[(index-1)*step_size:index * step_size]
            rows[:, :index * step_size] = 0
            mask[(index-1)*step_size:index * step_size] = rows
        
        return mask
    
    def encode_position(self, input1, input2):
        if self.add_day_feature:
            sequence_length = input2.size(-3)
            positional_values = torch.tensor([day_encoding/(sequence_length - 1) for day_encoding in range(0, sequence_length)], device=input1.device)
            new_feature_tensor = torch.zeros((*input2.shape[:-2], 1, 1), device=input1.device)
            new_feature_tensor[..., :, 0, 0] = positional_values

            repititions = [1 for each_ in new_feature_tensor.shape]
            repititions[-2] = input2.shape[-2]
            new_feature_tensor = new_feature_tensor.repeat(*repititions)
            
            input2 = torch.cat((input2, new_feature_tensor), -1)

        # original time2vec
        if self.positional_encoding == "time2vec":
            x = self.time2vec(input1)
            x_add = self.covariates_embedding(input2)
            x = x + x_add  
                        
        # simple linear layers
        elif self.positional_encoding == 'polar':
            #print(f"input1 shape before polar embedding: {input1.shape}")
            x = self.polar_embedding(input1)
            #print(f"x shape after polar embedding: {x.shape}")
            #print(f"input1 shape: {input1.shape}")
            x_add = self.polar_embedding(input2)
            x = x + x_add
            #print(f"input2 shape: {input2.shape}") 
            #print(f"x_add shape: {x_add.shape}")
        elif self.positional_encoding == "linear":
            x = self.input_encoding1(torch.cat((input1, input2), dim=-1))
            x = self.input_encoding2(x)
        # no encoding, simply concat tensors
        elif self.positional_encoding == None:
            x = torch.cat((input1, input2), dim=-1)
        else:
            raise ValueError("Input encoding not recognized.")
        
        return x


    def find_trailing_empty_slots(self, input_tensor):
            input_shape = input_tensor.size()
            ts_count = input_shape[-2]
            day_count = input_shape[-3]
            
            # Method 1, about 2-4x slower
            """
            TimeProfiler.begin("Slot finder Method 1")
            slot_count_1 = None
            checking_timeslot = 0
            found_last_value = False
            while found_last_value == False:
                cur_slice = x[..., pre_encoding_day_count - math.floor(checking_timeslot/pre_encoding_ts_count) - 1, pre_encoding_ts_count - (checking_timeslot % pre_encoding_ts_count) - 1, :]

                if torch.all(cur_slice == float(0)) == False:
                    print(f"Missing slots: {checking_timeslot}")
                    slot_count_1 = checking_timeslot
                    found_last_value = True
                checking_timeslot += 1
            TimeProfiler.end("Slot finder Method 1")
            """

            # Method 2, not functional
            """indices = torch.nonzero(x)[-1][-1].item() # possibly use this to remove empty values at the end of the tensor, potentially by increasing prediction horizon or by reordering to this: [empty][past][prediction] 
            last_nonzero_indices_var1 = torch.argmax(torch.flip((x == float(0)).flatten(start_dim=-3, end_dim=-2), dims=[-2]).int())
            print(f"Last non-zero1: {last_nonzero_indices_var1[-1] if len(last_nonzero_indices_var1.shape) > 1 and len(last_nonzero_indices_var1) > 0 else 'None found'}")
            """

            # Method 3
            last_nonzero_indices = torch.nonzero(input_tensor - self.empty_slot_value)
            if len(last_nonzero_indices) == 0: 
                raise ValueError("Entire input tensor is empty.")
            slot_count = (((day_count - 1) - last_nonzero_indices[-1][0].item()) * ts_count) + ((ts_count - 1) - last_nonzero_indices[-1][1].item())

            return slot_count
    
    def adjust_pos_weights(self, weights):
        self.pos_weight = torch.tensor([weights]).cuda() if torch.cuda.is_available() else torch.tensor([weights])
        self.loss_fun = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
    
    def configure_optimizers(self):
        # Transformer test notes:
        # LR_scheduler_max default: 0.0007 # suggested by LR finder: 0.03, 0.015, 0.00007; suggested by "Attention is all you need": 0.0007; the original paper for the OneCycleLR called "Super convergence: [...]" suggests very high values up to 6, but that likely doesn't translate 1:1 into PyTorch implementation
        # LR_scheduler_min default: 0.000000175 # suggested by "Attention is all you need": 0.000000175 
        # LR_scheduler_convergence default: 0.00016 # suggested by "Attention is all you need": degressivly decreasing value, lower than max_lr. This value is arbitrarily sampled from the formula when assuming that the total training duration is roughly 20 times as long as the warmup phase
        
        return model_util.configure_optimizers(self, self.use_optimizer, (self.optimizer_beta1, self.optimizer_beta2), self.weight_decay, self.use_LR_scheduler, self.LR_increase_phase_percentage, self.LR_scheduler_min, self.LR_scheduler_max, self.LR_scheduler_convergence, self.fixed_learning_rate)
