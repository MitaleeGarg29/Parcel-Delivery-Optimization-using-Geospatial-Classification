import torch
import pytorch_lightning as pl
import torch.nn as nn

from sklearn.metrics import precision_score, accuracy_score

from .internal.Time2Vec import Time2Vec
from .internal.model_util import model_util


class Transformer_V2_flatten(pl.LightningModule):
    """
    A transformer based model capable of performing predictions on a fixed prediction horizon. 
    The given data must be time-shifted within the Tensors containing the features which are not available in the future ("future unknowns").
    The prediction horizon is implicitly defined by the shift within the data.
    """
    def __init__(self, future_unknown_feature_count, future_known_covariate_feature_count, input_dim_3, add_day_feature=False, hidden_dim=10, window_size=1, use_mask="custom step", pos_weight=1, use_optimizer="Adam", optimizer_beta1=None, optimizer_beta2=None, weight_decay=None, fixed_learning_rate=0.001, log_LR=True, use_LR_scheduler=True, LR_scheduler_min=0.000000175, LR_scheduler_max=0.0007, LR_scheduler_convergence=0.00016, LR_increase_phase_percentage=0.05):
        super(Transformer_V2_flatten, self).__init__()

        extra_dim = 0
        if add_day_feature == True: extra_dim = 1

        self.time2vec = Time2Vec(future_unknown_feature_count, hidden_dim)
        self.transformer = nn.Transformer(hidden_dim * 2, 4, batch_first=True)
        self.weather_embedding = nn.Linear(future_known_covariate_feature_count + extra_dim, hidden_dim*2)
        self.pos_weight = torch.tensor([pos_weight]).cuda() if torch.cuda.is_available() else torch.tensor([pos_weight])
        self.loss_fun = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.fc = nn.Linear(hidden_dim*2, 1)

        self.use_mask = use_mask
        self.add_day_feature = add_day_feature

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

        model_util.configure_logging(self, log_lr = self.log_LR)
        self.save_hyperparameters()


    def forward(self, x, x_add=None, batch_idx=None):
        if x_add == None: 
            x, x_add, label = x

        if self.add_day_feature == True:
            sequence_length = x_add.size(-3)
            positional_values = torch.tensor([day_encoding/(sequence_length - 1) for day_encoding in range(0, sequence_length)], device=x_add.device)
            new_feature_tensor = torch.zeros((*x_add.shape[:-2], 1, 1), device=x_add.device)
            new_feature_tensor[..., :, 0, 0] = positional_values

            repititions = [1 for each_ in new_feature_tensor.shape]
            repititions[-2] = x_add.shape[-2]
            new_feature_tensor = new_feature_tensor.repeat(*repititions)
            
            x_add = torch.cat((x_add, new_feature_tensor), -1)

        x = self.time2vec(x)
        x_add = self.weather_embedding(x_add)
        x = x + x_add                

        prev_size = x.size()    # [batch, day, time, feature] or [day, time, feature] 
        x = x.flatten(-3, -2)   # [batch, day x time, feature] or [day x time, feature]    


        attention_mask = None
        if self.use_mask == "subsequent": 
            attention_mask = nn.Transformer.generate_square_subsequent_mask(x.shape[-2], self.device)
        elif self.use_mask == "custom step":
            attention_mask = self.generate_custom_causal_mask(x.shape[-2], prev_size[-2], self.device)
        else:
            raise ValueError("Mask type not supported.")

        output = self.transformer(x, x, src_mask=attention_mask, tgt_mask=attention_mask, src_is_causal=True, tgt_is_causal=True)
        output = output.view(prev_size)
        output = self.fc(output) 

        return output


    def training_step(self, batch, batch_idx):
        x, x_add, target = batch
        pred = self(x, x_add, batch_idx)

        mask = ~target.isnan()

        filtered_pred = pred.squeeze(len(pred.shape)-1)[mask]
        filtered_target = target[mask]

        loss = self.loss_fun(filtered_pred, filtered_target)

        binary_pred = (torch.sigmoid(pred) > 0.5).int()
        filtered_binary_pred = binary_pred[mask]

        model_util.log(self, loss, filtered_binary_pred, filtered_target, "train")
        model_util.batch_step(self, self.log_LR, self.use_LR_scheduler)

        return loss


    def validation_step(self, batch, batch_idx):
        x, x_add, target = batch
        pred = self(x, x_add)

        mask = ~target.isnan()

        filtered_pred = pred.squeeze(len(pred.shape)-1)[mask]
        filtered_target = target[mask]

        loss = self.loss_fun(filtered_pred, filtered_target)

        binary_pred = (torch.sigmoid(pred) > 0.5).int()
        filtered_binary_pred = binary_pred[mask]
        
        model_util.log(self, loss, filtered_binary_pred, filtered_target, "val")

        return {'val_loss': loss}

    
    def configure_optimizers(self):
        # Transformer test notes:
        # LR_scheduler_max default: 0.0007 # suggested by LR finder: 0.03, 0.015, 0.00007; suggested by "Attention is all you need": 0.0007; the original paper for the OneCycleLR called "Super convergence: [...]" suggests very high values up to 6, but that likely doesn't translate 1:1 into PyTorch implementation
        # LR_scheduler_min default: 0.000000175 # suggested by "Attention is all you need": 0.000000175 
        # LR_scheduler_convergence default: 0.00016 # suggested by "Attention is all you need": degressivly decreasing value, lower than max_lr. This value is arbitrarily sampled from the formula when assuming that the total training duration is roughly 20 times as long as the warmup phase
        
        return model_util.configure_optimizers(self, self.use_optimizer, (self.optimizer_beta1, self.optimizer_beta2), self.weight_decay, self.use_LR_scheduler, self.LR_increase_phase_percentage, self.LR_scheduler_min, self.LR_scheduler_max, self.LR_scheduler_convergence, self.fixed_learning_rate)

    # creates a mask like torch.triu, but allows steps larger than 1
    def generate_custom_causal_mask(self, size, step_size, device):
        mask = torch.full((size, size), float('-inf'), device=device)
        for index in range(1, size):
            rows = mask[(index-1)*step_size:index * step_size]
            rows[:, :index * step_size] = 0
            mask[(index-1)*step_size:index * step_size] = rows
        
        return mask
            
    def adjust_pos_weights(self, weights):
        self.pos_weight = torch.tensor([weights]).cuda() if torch.cuda.is_available() else torch.tensor([weights])
        self.loss_fun = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
