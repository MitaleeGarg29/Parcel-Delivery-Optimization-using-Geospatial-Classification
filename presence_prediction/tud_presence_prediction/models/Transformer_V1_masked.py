import torch
import pytorch_lightning as pl
import torch.nn as nn

from sklearn.metrics import precision_score, accuracy_score
from .internal.Time2Vec import Time2Vec
from .internal.model_util import model_util

class Transformer_V1_masked(pl.LightningModule):
    def __init__(self, in_dim, weather_dim, input_dim_3, hidden_dim=10, window_size=1):
        super(Transformer_V1_masked, self).__init__()
        self.time2vec = Time2Vec(in_dim, hidden_dim)
        #self.time2vec=time2vec.Model(activation="sin",hidden_dim=hidden_dim)
        # TODO: Figure out how torch handles multiple dimensions
        self.transformer = nn.Transformer(hidden_dim * 2, 4)
        self.weather_embedding = nn.Linear(weather_dim, hidden_dim*2)
        self.pos_weight = torch.tensor([2 / 10]).cuda() if torch.cuda.is_available() else torch.tensor([2 / 10])
        self.loss_fun = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.fc = nn.Linear(hidden_dim*2, 1)

        model_util.configure_logging(self, False)
        self.save_hyperparameters()


    def forward(self, x, x_add=None):
        if x_add == None: 
            x, x_add, label = x

        # Reshaping from (days, times, features) to (days*times, features)
        x = x.view(-1, x.size(-1))
        x_add = x_add.view(-1, x_add.size(-1))

        # Generate the mask for the flattened sequence
        sequence_length = x.size(0)
        mask = self.generate_square_subsequent_mask(sequence_length).to(x.device)

        x = self.time2vec(x)
        x_add = self.weather_embedding(x_add)
        x = x + x_add

        output = self.transformer(x, x, src_mask=mask)
        output = self.fc(output) 

        return output


    def training_step(self, batch, batch_idx):
        x, x_add, target = batch

        pred = self(x, x_add)

        target = target.view(-1)
        mask = ~target.isnan()

        filtered_pred = pred.squeeze(len(pred.shape)-1)[mask]
        filtered_target = target[mask]

        loss = self.loss_fun(filtered_pred, filtered_target)

        binary_pred = (pred > 0.5).type(torch.IntTensor) # TODO: replace with filtered_pred?
        filtered_binary_pred = binary_pred[mask]

        model_util.log(self, loss, filtered_binary_pred, filtered_target, "train")
        model_util.batch_step(self, False, False)

        return loss


    def validation_step(self, batch, batch_idx):
        x, x_add, target = batch

        pred = self(x, x_add)

        target = target.view(-1)
        mask = ~target.isnan()

        filtered_pred = pred.squeeze(len(pred.shape)-1)[mask]
        filtered_target = target[mask]

        loss = self.loss_fun(filtered_pred, filtered_target)

        binary_pred = (pred > 0.5).type(torch.IntTensor)    # TODO: replace with filtered_pred?
        filtered_binary_pred = binary_pred[mask]
        
        model_util.log(self, loss, filtered_binary_pred, filtered_target, "val")

        return {'val_loss': loss}

    
    def configure_optimizers(self):
        return model_util.configure_optimizers(self, "Adam")

    
    
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(
            torch.ones(sz, sz).bool(), 
            diagonal=1
        )

        return mask
