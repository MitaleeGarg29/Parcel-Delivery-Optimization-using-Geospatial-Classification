import torch
import pytorch_lightning as pl

from .internal.pew_lstm_longmem import Pew_LSTM_longmem
from sklearn.metrics import precision_score, accuracy_score

from helpers.profiling.time_profiling import TimeProfiler
from .internal.model_util import model_util

# Expects data of type (input, input, label)
# Expects parameters of type (input, input)
class LSTM_V1_longmem(pl.LightningModule):
    def __init__(self, in_dim, weather_dim, input_dim_3, hidden_dim=10, window_size=1, pos_weight=1, use_optimizer="Adamax", fixed_learning_rate=0.001, log_LR=True, use_LR_scheduler=True, LR_scheduler_min=0.02, LR_scheduler_max=0.03, LR_scheduler_convergence=0.005, LR_increase_phase_percentage=0.6, optimizer_beta1=0.51, optimizer_beta2=0.999):
        super(LSTM_V1_longmem, self).__init__()
        self.lstm = Pew_LSTM_longmem(in_dim=in_dim, hidden_dim=hidden_dim, weather_dim=weather_dim, window_size=window_size)
        self.pos_weight = torch.tensor([pos_weight]).cuda() if torch.cuda.is_available() else torch.tensor([pos_weight])
        self.loss_fun = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        
        self.use_optimizer = use_optimizer
        self.optimizer_beta1 = optimizer_beta1
        self.optimizer_beta2 = optimizer_beta2
        self.fixed_learning_rate = fixed_learning_rate
        self.log_LR = log_LR
        self.use_LR_scheduler = use_LR_scheduler
        self.LR_scheduler_min = LR_scheduler_min
        self.LR_scheduler_max = LR_scheduler_max
        self.LR_scheduler_convergence = LR_scheduler_convergence
        self.LR_increase_phase_percentage = LR_increase_phase_percentage

        model_util.configure_logging(self, log_lr = self.log_LR)
        self.save_hyperparameters()


    def forward(self, x, x_add=None):
        if x_add == None: 
            x, x_add, label = x
        return self.lstm(x, x_add)

    def training_step(self, batch, batch_idx=None, dataloader_idx=None):
        TimeProfiler.begin("LSTM Training_step")
        x, x_add, target = batch
        TimeProfiler.begin("LSTM Forward")
        pred = self(x, x_add)
        TimeProfiler.end("LSTM Forward")

        mask = ~target.isnan()
        
        filtered_pred = pred[mask]
        filtered_target = target[mask]

        TimeProfiler.begin("LSTM Loss Fun")
        loss = self.loss_fun(filtered_pred, filtered_target)
        TimeProfiler.end("LSTM Loss Fun")
        
        binary_pred = (pred > 0.5).int()
        filtered_binary_pred = binary_pred[mask]

        model_util.log(self, loss, filtered_binary_pred, filtered_target, "train")
        model_util.batch_step(self, self.log_LR, self.use_LR_scheduler)

        TimeProfiler.end("LSTM Training_step")

        return loss

    

    def validation_step(self, batch, batch_idx=None, dataloader_idx=None):
        x, x_add, target = batch
        pred = self(x, x_add)

        mask = ~target.isnan()

        filtered_pred = pred[mask]
        filtered_target = target[mask]

        loss = self.loss_fun(filtered_pred, filtered_target)
        
        binary_pred = (pred > 0.5).int()
        filtered_binary_pred = binary_pred[mask]
        
        model_util.log(self, loss, filtered_binary_pred, filtered_target, "val")

        return {'val_loss': loss}

        # TODO: RETURN VALUE? https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    
    
    def configure_optimizers(self):
        return model_util.configure_optimizers(self, self.use_optimizer, (self.optimizer_beta1, self.optimizer_beta2), None, self.use_LR_scheduler, self.LR_increase_phase_percentage, self.LR_scheduler_min, self.LR_scheduler_max, self.LR_scheduler_convergence, self.fixed_learning_rate)

        
    def adjust_pos_weights(self, weights):
        self.pos_weight = torch.tensor([weights]).cuda() if torch.cuda.is_available() else torch.tensor([weights])
        self.loss_fun = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    #def on_validation_epoch_end(self):
    #    super().on_validation_epoch_end()
    #    print("")
    #    print("")
    #    print(self.validation_step_outputs[-1])
    #    print("")
        

    #def on_train_epoch_end(self) -> None:
    #    super().on_train_epoch_end()
    #    print("")
    #    print("")
    #    print(self.train_step_outputs[-1])
    #    print("")