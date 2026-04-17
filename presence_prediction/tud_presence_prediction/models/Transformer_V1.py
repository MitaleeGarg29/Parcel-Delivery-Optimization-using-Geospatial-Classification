import torch
import pytorch_lightning as pl

import torch.nn as nn
from sklearn.metrics import precision_score, accuracy_score

from .internal.Time2Vec import Time2Vec

class Transformer_V1(pl.LightningModule):
    def __init__(self, in_dim, weather_dim, input_dim_3, hidden_dim=10, window_size=1):
        super(Transformer_V1, self).__init__()
        self.time2vec = Time2Vec(in_dim, hidden_dim)
        #self.time2vec=time2vec.Model(activation="sin",hidden_dim=hidden_dim)
        self.transformer = nn.Transformer(hidden_dim * 2, 4)
        self.weather_embedding = nn.Linear(weather_dim, hidden_dim*2)
        self.pos_weight = torch.tensor([2 / 10]).cuda() if torch.cuda.is_available() else torch.tensor([2 / 10])
        self.loss_fun = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        # TODO: create softmax layer with enough timeslots for one day
        self.train_losses = []
        self.val_losses = []
        self.fc = nn.Linear(hidden_dim*2, 1)
        self.train_accuracies = []
        self.train_precisions = []
        self.val_accuracies = []
        self.val_precisions = []

        self.visual_metrics = dict()
        self.visual_metrics["Training loss"] = []
        self.visual_metrics["Training precision"] = []
        self.visual_metrics["Training accuracy"] = []

        self.visual_metrics["Validation loss"] = []

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

        mask = ~target.isnan().squeeze()

        t1 = pred.squeeze()[mask]
        t2 = target.squeeze()[mask]

        loss = self.loss_fun(t1, t2)
        self.log('train_loss', loss)
        self.train_losses.append(loss.item())
        binary_pred = (pred > 0.5).type(torch.IntTensor)

        t3 = target[~target.isnan()].detach().cpu().numpy()
        t4 = binary_pred[~target.isnan()].detach().cpu().numpy()

        precision = float(0)
        accuracy = float(0)
        
        if t3.size != 0: 
            precision = precision_score(t3, t4)
            accuracy = accuracy_score(target[~target.isnan()], binary_pred[~target.isnan()])

        self.log('train_precision', precision)
        self.log('train_accuracy', accuracy)
        self.train_accuracies.append(accuracy)
        self.train_precisions.append(precision)

        self.visual_metrics["Training loss"].append(loss.item())

        self.visual_metrics["Training precision"].append(precision)
        self.visual_metrics["Training accuracy"].append(accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        x, x_add, target = batch
        pred = self(x, x_add)
        mask = ~target.isnan()
        loss = self.loss_fun(pred.squeeze()[mask], target[mask])
        self.log('val_loss', loss)
        self.val_losses.append(loss.item())
        binary_pred = (pred > 0.5).type(torch.IntTensor)
        precision = precision_score(target[~target.isnan()].detach().cpu().numpy(),
                                    binary_pred[~target.isnan()].detach().cpu().numpy())
        accuracy = accuracy_score(target[~target.isnan()], binary_pred[~target.isnan()])
        self.log('val_precision', precision)
        self.log('val_accuracy', accuracy)
        self.val_accuracies.append(accuracy)
        self.val_precisions.append(precision)

        self.visual_metrics["Validation loss"].append(loss.item())

        #self.visual_metrics["Validation precision"].append(precision)
        #self.visual_metrics["Validation accuracy"].append(accuracy)

        return {'val_loss': loss}

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())