import torch
import pytorch_lightning as pl

from .internal.pew_lstm_shift_longmem import Pew_LSTM_shift_longmem
from sklearn.metrics import precision_score, accuracy_score
from .internal.model_util import model_util


# Expects data of type (input, input, label)
# Expects parameters of type (input, input)
class LSTM_V1_Shift_Longmem(pl.LightningModule):
    def __init__(self, in_dim, weather_dim, input_dim_3, hidden_dim=10, window_size=1, pos_weight=1, day_shift=1):
        super(LSTM_V1_Shift_Longmem, self).__init__()

        self.day_shift = day_shift

        self.lstm = Pew_LSTM_shift_longmem(in_dim=in_dim, hidden_dim=hidden_dim, weather_dim=weather_dim, window_size=window_size, day_shift=day_shift)
        self.pos_weight = torch.tensor([pos_weight]).cuda() if torch.cuda.is_available() else torch.tensor([pos_weight])
        self.loss_fun = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        model_util.configure_logging(self, False)
        self.save_hyperparameters() 


    def forward(self, x, x_add=None):
        if x_add == None: 
            x, x_add, label = x
        return self.lstm(x, x_add)

    def training_step(self, batch, batch_idx):
        x, x_add, target = batch
        pred = self(x, x_add)

        mask = ~target.isnan()
        mask[0:self.day_shift][0:73] = False
        
        filtered_pred = pred[mask]
        filtered_target = target[mask]

        loss = self.loss_fun(filtered_pred, filtered_target)
        
        binary_pred = (pred > 0.5).type(torch.IntTensor)    # TODO: replace with filtered_pred?
        filtered_binary_pred = binary_pred[mask]
        
        model_util.log(self, loss, filtered_binary_pred, filtered_target, "train")
        model_util.batch_step(self, False, False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, x_add, target = batch
        pred = self(x, x_add)

        mask = ~target.isnan()
        mask[0:self.day_shift][0:73] = False

        filtered_pred = pred[mask]
        filtered_target = target[mask]

        loss = self.loss_fun(filtered_pred, filtered_target)
        
        binary_pred = (pred > 0.5).type(torch.IntTensor)    # TODO: replace with filtered_pred?
        filtered_binary_pred = binary_pred[mask]
        
        model_util.log(self, loss, filtered_binary_pred, filtered_target, "val")

        return {'val_loss': loss}
    
    def configure_optimizers(self):
        return model_util.configure_optimizers(self, "Adam")

