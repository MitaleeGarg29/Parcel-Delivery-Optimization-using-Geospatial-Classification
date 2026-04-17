import torch
import pytorch_lightning as pl

import torch.nn as nn
from sklearn.metrics import precision_score, accuracy_score

from .internal.Time2Vec import Time2Vec

class Transformer_V1_full_fix(pl.LightningModule):
    def __init__(self, in_dim, weather_dim, input_dim_3, hidden_dim=10, window_size=1):
        super(Transformer_V1_full_fix, self).__init__()
        self.time2vec = Time2Vec(in_dim, hidden_dim)
        #self.time2vec=time2vec.Model(activation="sin",hidden_dim=hidden_dim)
        self.transformer = nn.Transformer(hidden_dim * 2, 4)
        self.weather_embedding = nn.Linear(weather_dim, hidden_dim*2)
        self.pos_weight = torch.tensor([2 / 10]).cuda() if torch.cuda.is_available() else torch.tensor([2 / 10])
        self.loss_fun = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        # TODO: create softmax layer with enough timeslots for one day
        self.fc = nn.Linear(hidden_dim*2, 1)


        self.visual_metrics = dict()
        self.visual_metrics["Training loss"] = []
        self.visual_metrics["Training precision"] = []
        self.visual_metrics["Training accuracy"] = []
        self.visual_metrics["Validation loss"] = []
        self.visual_metrics["Validation precision"] = []
        self.visual_metrics["Validation accuracy"] = []

        self.save_hyperparameters()


    def forward(self, x, x_add):
        x = self.time2vec(x)
        x_add = self.weather_embedding(x_add)
        x = x + x_add
        output = self.transformer(x,x)
        output = self.fc(output)  # Add this line
        return output

    def training_step(self, batch, batch_idx):
        x, x_add, target = batch
        pred = self(x, x_add)

        mask = ~target.isnan()

        filtered_pred = pred.squeeze(len(pred.shape)-1)[mask]
        filtered_target = target[mask]

        loss = self.loss_fun(filtered_pred, filtered_target)

        binary_pred = (pred > 0.5).type(torch.IntTensor) # TODO: replace with filtered_pred?
        filtered_binary_pred = binary_pred[mask]

        filtered_numpy_target = filtered_target.detach().cpu().numpy()
        filtered_numpy_binary_pred = filtered_binary_pred.detach().cpu().numpy()

        precision = precision_score(filtered_numpy_target, filtered_numpy_binary_pred, zero_division=0)
        accuracy = accuracy_score(filtered_target, filtered_numpy_binary_pred)
        
        self.log('train_loss', loss)
        self.log('train_precision', precision)
        self.log('train_accuracy', accuracy)

        self.visual_metrics["Training loss"].append(loss.item())
        self.visual_metrics["Training precision"].append(precision)
        self.visual_metrics["Training accuracy"].append(accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        x, x_add, target = batch
        pred = self(x, x_add)

        mask = ~target.isnan()

        filtered_pred = pred.squeeze(len(pred.shape)-1)[mask]
        filtered_target = target[mask]

        loss = self.loss_fun(filtered_pred, filtered_target)

        binary_pred = (pred > 0.5).type(torch.IntTensor)    # TODO: replace with filtered_pred?
        filtered_binary_pred = binary_pred[mask]
        
        filtered_numpy_target = filtered_target.detach().cpu().numpy()
        filtered_numpy_binary_pred = filtered_binary_pred.detach().cpu().numpy()

        precision = precision_score(filtered_numpy_target, filtered_numpy_binary_pred, zero_division=0)
        accuracy = accuracy_score(filtered_numpy_target, filtered_numpy_binary_pred)
        
        self.log('val_loss', loss)
        self.log('val_precision', precision)
        self.log('val_accuracy', accuracy)

        self.visual_metrics["Validation loss"].append(loss.item())
        self.visual_metrics["Validation precision"].append(precision)
        self.visual_metrics["Validation accuracy"].append(accuracy)

        return {'val_loss': loss}

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())