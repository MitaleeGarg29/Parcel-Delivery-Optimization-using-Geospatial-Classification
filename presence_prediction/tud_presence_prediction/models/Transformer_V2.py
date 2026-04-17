import torch
import pytorch_lightning as pl
import torch.nn as nn

from sklearn.metrics import precision_score, accuracy_score
from .internal.Time2Vec import Time2Vec
from .internal.model_util import model_util
from helpers.profiling.time_profiling import TimeProfiler

class Transformer_V2(pl.LightningModule):
    def __init__(self, in_dim, weather_dim, input_dim_3, hidden_dim=10, window_size=1, pos_weight=1, batch_first_dim=False, attention_dim_first=True, use_optimizer="Adam", optimizer_beta1=None, optimizer_beta2=None, weight_decay=None, fixed_learning_rate=0.001, log_LR=True, use_LR_scheduler=True, LR_scheduler_min=0.000000175, LR_scheduler_max=0.0007, LR_scheduler_convergence=0.00016, LR_increase_phase_percentage=0.05):
        super(Transformer_V2, self).__init__()
        self.time2vec = Time2Vec(in_dim, hidden_dim)
        #self.time2vec=time2vec.Model(activation="sin",hidden_dim=hidden_dim)
        self.transformer = nn.Transformer(hidden_dim * 2, 4, batch_first=batch_first_dim)
        self.weather_embedding = nn.Linear(weather_dim, hidden_dim*2)
        self.pos_weight = torch.tensor([pos_weight]).cuda() if torch.cuda.is_available() else torch.tensor([pos_weight])
        self.loss_fun = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.fc = nn.Linear(hidden_dim*2, 1)

        self.attention_dim_first = attention_dim_first
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

        # Generate the mask for the flattened sequence
        #sequence_length = x.size(0)
        #mask = self.generate_square_subsequent_mask(sequence_length).to(x.device)

        #print(f"Batch {batch_idx} Shape: " + str(x.shape) + " and " + str(x_add.shape))

        x = self.time2vec(x)
        x_add = self.weather_embedding(x_add)
        x = x + x_add

        attention_mask = nn.Transformer.generate_square_subsequent_mask(x.shape[0 if self.attention_dim_first == True else 1], self.device)
        output = self.transformer(x, x, src_mask=attention_mask, tgt_mask=attention_mask, src_is_causal=True, tgt_is_causal=True)
        output = self.fc(output) 

        return output


    def training_step(self, batch, batch_idx):
        TimeProfiler.begin("Transformer Training Step")
        x, x_add, target = batch
        TimeProfiler.begin("Transformer Forward")
        pred = self(x, x_add, batch_idx)
        TimeProfiler.end("Transformer Forward")

        TimeProfiler.begin("Transformer Filtering")
        mask = ~target.isnan()

        filtered_pred = pred.squeeze(len(pred.shape)-1)[mask]
        filtered_target = target[mask]
        TimeProfiler.end("Transformer Filtering")

        TimeProfiler.begin("Transformer Loss Fun")
        loss = self.loss_fun(filtered_pred, filtered_target)
        TimeProfiler.end("Transformer Loss Fun")

        TimeProfiler.begin("Transformer Binary Pred")
        binary_pred = (pred > 0.5).int() #.type(torch.IntTensor)

        filtered_binary_pred = binary_pred[mask]
        TimeProfiler.end("Transformer Binary Pred")

        model_util.log(self, loss, filtered_binary_pred, filtered_target, "train")
        model_util.batch_step(self, self.log_LR, self.use_LR_scheduler)

        #print(f"TRAINING: Batch {batch_idx} Shape: " + str(x.shape) + " and " + str(x_add.shape))
        TimeProfiler.end("Transformer Training Step")
        return loss


    def validation_step(self, batch, batch_idx):
        x, x_add, target = batch
        pred = self(x, x_add)

        mask = ~target.isnan()

        filtered_pred = pred.squeeze(len(pred.shape)-1)[mask]
        filtered_target = target[mask]

        loss = self.loss_fun(filtered_pred, filtered_target)

        binary_pred = (pred > 0.5).int() #.type(torch.IntTensor)
        filtered_binary_pred = binary_pred[mask]
        
        model_util.log(self, loss, filtered_binary_pred, filtered_target, "val")

        return {'val_loss': loss}

    
    def configure_optimizers(self):
        # Transformer test notes:
        # LR_scheduler_max default: 0.0007 # suggested by LR finder: 0.03, 0.015, 0.00007; suggested by "Attention is all you need": 0.0007; the original paper for the OneCycleLR called "Super convergence: [...]" suggests very high values up to 6, but that likely doesn't translate 1:1 into PyTorch implementation
        # LR_scheduler_min default: 0.000000175 # suggested by "Attention is all you need": 0.000000175 
        # LR_scheduler_convergence default: 0.00016 # suggested by "Attention is all you need": degressivly decreasing value, lower than max_lr. This value is arbitrarily sampled from the formula when assuming that the total training duration is roughly 20 times as long as the warmup phase
        
        return model_util.configure_optimizers(self, self.use_optimizer, (self.optimizer_beta1, self.optimizer_beta2), self.weight_decay, self.use_LR_scheduler, self.LR_increase_phase_percentage, self.LR_scheduler_min, self.LR_scheduler_max, self.LR_scheduler_convergence, self.fixed_learning_rate)

    
    #def generate_square_subsequent_mask(self, sz):
    #    mask = torch.triu(
    #        torch.ones(sz, sz).bool(), 
    #        diagonal=1
    #    )

    #    return mask
