import torch
import torch.nn as nn
import math

from torch.nn import Parameter
from torch.nn import init
from torch import Tensor
from .utils.model_utils import window_view

from tud_presence_prediction.helpers.profiling.time_profiling import TimeProfiler

class Pew_LSTM_layer_longmem_batch(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, weather_size: int):
        super(Pew_LSTM_layer_longmem_batch, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weather_size = weather_size

        # input gate
        self.w_ix = Parameter(Tensor(hidden_size, input_size))
        self.w_ih = Parameter(Tensor(hidden_size, hidden_size))
        self.w_ie = Parameter(Tensor(hidden_size, hidden_size))  # this vector in paper is "w_fe"
        self.b_i = Parameter(Tensor(hidden_size, 1))

        # forget gate
        self.w_fx = Parameter(Tensor(hidden_size, input_size))
        self.w_fo = Parameter(Tensor(hidden_size, hidden_size))
        self.w_fe = Parameter(Tensor(hidden_size, hidden_size))
        self.b_f = Parameter(Tensor(hidden_size, 1))

        # output gate
        self.w_ox = Parameter(Tensor(hidden_size, input_size))
        self.w_oh = Parameter(Tensor(hidden_size, hidden_size))
        self.w_oe = Parameter(Tensor(hidden_size, hidden_size))
        self.b_o = Parameter(Tensor(hidden_size, 1))

        # cell
        self.w_gx = Parameter(Tensor(hidden_size, input_size))
        self.w_gh = Parameter(Tensor(hidden_size, hidden_size))
        self.b_g = Parameter(Tensor(hidden_size, 1))

        # ho
        self.w_d = Parameter(Tensor(hidden_size, input_size))
        self.w_w = Parameter(Tensor(hidden_size, input_size))
        self.w_m = Parameter(Tensor(hidden_size, input_size))
        self.w_t = Parameter(Tensor(hidden_size, hidden_size))
        self.w_e = Parameter(Tensor(hidden_size, weather_size))
        self.b_e = Parameter(Tensor(hidden_size, 1))  # this vector in paper is "b_f"

        self.reset_weigths()

    def reset_weigths(self):
        """reset weights
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, x_input, x_weather):
        """Forward
        Args:
            inputs: [batch_size, seq_size, input_size]
            weathers: [batch_size, seq_size, weather_size]
        """
        using_batches = False
        day_size, seq_size, input_dim = x_input.size()[-3:] 
        if len(x_input.size()) > 3: 
            using_batches = True
            batch_size, day_size, seq_size, input_dim = x_input.size()
        

        TimeProfiler.begin("LSTM Memory Setup")
        output_size = (day_size, seq_size, self.hidden_size)
        if using_batches: output_size = (batch_size, day_size, seq_size, self.hidden_size)
        h_output = torch.zeros(output_size, device=x_input.device)
        c_output = torch.zeros(output_size, device=x_input.device)

        memory_size = (1, self.hidden_size)
        if using_batches: memory_size = (batch_size, 1, self.hidden_size)
        h_t = torch.transpose(torch.zeros(memory_size, device=x_input.device), -2, -1)
        c_t = torch.transpose(torch.zeros(memory_size, device=x_input.device), -2, -1)
        TimeProfiler.end("LSTM Memory Setup")

        replacement_size = (1, self.input_size)
        if using_batches: replacement_size = (batch_size, 1, self.input_size)

        unsqueeze_dim = 0
        if using_batches: unsqueeze_dim = 1

        for d in range(day_size):
            #h_t = torch.zeros(1, self.hidden_size,device=x_input.device).t()
            #c_t = torch.zeros(1, self.hidden_size,device=x_input.device).t()

            
            for t in range(seq_size):
                #print(f"DAY COUNT: {d}")
                TimeProfiler.begin("LSTM PastVar Setup")
                # day
                if d < 1:
                    h_d = torch.transpose(torch.zeros(replacement_size, device=x_input.device), -2, -1)
                else:
                    h_d = torch.transpose(x_input[..., d - 1, t, :].unsqueeze(unsqueeze_dim), -2, -1)

                # week
                if d < 7:
                    h_w = torch.transpose(torch.zeros(replacement_size, device=x_input.device), -2, -1)
                else:
                    h_w = torch.transpose(x_input[..., d - 6, t, :].unsqueeze(unsqueeze_dim), -2, -1)

                # month
                if d < 14 * 2:
                    h_m = torch.transpose(torch.zeros(replacement_size, device=x_input.device), -2, -1)
                else:
                    h_m = torch.transpose(x_input[..., d - 29, t, :].unsqueeze(unsqueeze_dim), -2, -1)

                TimeProfiler.end("LSTM PastVar Setup")

                TimeProfiler.begin("LSTM Input assignment")
                x = torch.transpose(x_input[..., d, t, :].unsqueeze(unsqueeze_dim), -2, -1)  # [input_dim, 1]
                weather_t = torch.transpose(x_weather[..., d, t, :].unsqueeze(unsqueeze_dim), -2, -1)  # [weather_dim, 1]
                TimeProfiler.end("LSTM Input assignment")

                """
                TimeProfiler.begin("LSTM Calc Old")
                # replace h_t with ho
                h_o = torch.sigmoid(self.w_d @ h_d + self.w_w @ h_w + self.w_t @ h_t +
                                    self.w_m @ h_m + self.w_t @ h_t)
                e_t = torch.sigmoid(self.w_e @ weather_t + self.b_e)
                # input gate
                i = torch.sigmoid(self.w_ix @ x + self.w_ih @ h_o +
                                  self.w_ie @ e_t + self.b_i)
                # cell
                g = torch.tanh(self.w_gx @ x + self.w_gh @ h_o
                               + self.b_g)
                # forget gate
                f = torch.sigmoid(self.w_fx @ x + self.w_fo @ h_o +
                                  self.w_fe @ e_t + self.b_f)

                # output gate
                o = torch.sigmoid(self.w_ox @ x + self.w_oh @ h_t +
                                  self.w_oe @ e_t + self.b_o)

                c_next = f * c_t + i * g  # [hidden_dim, 1]
                h_next = o * torch.tanh(c_next)  # [hidden_dim, 1]

                TimeProfiler.end("LSTM Calc Old")
                """
                
                """
                print("")
                print("Batch check:")
                print(x.shape)
                print(weather_t.shape)
                print(h_d.shape)
                print(h_w.shape)
                print(h_m.shape)
                print(h_t.shape)
                print(self.w_d.shape)
                print(self.w_m.shape)
                print(self.w_t.shape)
                print(self.w_w.shape)
                print(self.w_e.shape)
                print(self.w_ix.shape)
                print(self.w_ih.shape)
                print(self.w_ie.shape)
                print(self.w_gx.shape)
                print(self.w_gh.shape)
                print(self.w_fx.shape)
                print(self.w_fo.shape)
                print(self.w_fe.shape)
                print(self.w_ox.shape)
                print(self.w_oh.shape)
                print(self.w_oe.shape)
                """



                TimeProfiler.begin("LSTM Calc new")
                h_o = torch.sigmoid(
                    torch.matmul(self.w_d, h_d) + 
                    torch.matmul(self.w_w, h_w) + 
                    torch.matmul(self.w_t, h_t) +
                    torch.matmul(self.w_m, h_m) + 
                    torch.matmul(self.w_t, h_t)
                )
                e_t = torch.sigmoid(
                    torch.matmul(self.w_e,weather_t) + 
                    self.b_e
                )
                i = torch.sigmoid(
                    torch.matmul(self.w_ix, x) + 
                    torch.matmul(self.w_ih, h_o) +
                    torch.matmul(self.w_ie, e_t) + 
                    self.b_i
                )
                g = torch.tanh(
                    torch.matmul(self.w_gx, x) + 
                    torch.matmul(self.w_gh, h_o) + 
                    self.b_g
                )
                f = torch.sigmoid(
                    torch.matmul(self.w_fx, x) + 
                    torch.matmul(self.w_fo, h_o) +
                    torch.matmul(self.w_fe, e_t) + 
                    self.b_f
                )
                o = torch.sigmoid(
                    torch.matmul(self.w_ox, x) + 
                    torch.matmul(self.w_oh, h_t) +
                    torch.matmul(self.w_oe, e_t) + 
                    self.b_o
                )

                #print(h_o.shape)
                #print(e_t.shape)

                c_next = f * c_t + i * g
                h_next = o * torch.tanh(c_next)
                TimeProfiler.end("LSTM Calc new")

                TimeProfiler.begin("LSTM Squeeze and T")
                if using_batches:
                    h_output[:, d, t] = torch.transpose(h_next, -2, -1).squeeze(unsqueeze_dim)
                    c_output[:, d, t] = torch.transpose(c_next, -2, -1).squeeze(unsqueeze_dim)
                else:
                    h_output[d, t] = torch.transpose(h_next, -2, -1).squeeze(unsqueeze_dim)
                    c_output[d, t] = torch.transpose(c_next, -2, -1).squeeze(unsqueeze_dim)
                TimeProfiler.end("LSTM Squeeze and T")

                h_t = h_next
                c_t = c_next

        return (h_output, c_output)


HIDDEN_DIM = 10


class Pew_LSTM_longmem_batch(nn.Module):
    # timemode: 0 for day, 1 for week, 2 for
    def __init__(self, in_dim: int, hidden_dim: int, weather_dim: int, window_size: int = 30):
        super(Pew_LSTM_longmem_batch, self).__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.lstm1 = Pew_LSTM_layer_longmem_batch(in_dim * window_size, hidden_dim, weather_dim * window_size)
        self.lstm2 = Pew_LSTM_layer_longmem_batch(hidden_dim, hidden_dim, weather_dim * window_size)
        self.fc = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x_input, x_weather):  # [batch_size, seq_size, weather_size + input_dim]
        # x_weather = input[:, :, :-1]  # [batch_size, seq_size, weather_size]
        # x_input = input[:, :, -9].unsqueeze(2)  # [batch_size, seq_size, input_dim]

        TimeProfiler.begin("LSTM w_view_test")
        w_view_test = window_view(x_weather,self.window_size)
        TimeProfiler.end("LSTM w_view_test")

        TimeProfiler.begin("LSTM L1 Forward")
        h1, c1 = self.lstm1(window_view(x_input,self.window_size),
                            window_view(x_weather,self.window_size))  # ([batch_size, seq_size, hidden_size], [batch_size, seq_size, hidden_size])
        TimeProfiler.end("LSTM L1 Forward")

        TimeProfiler.begin("LSTM L2 Forward")
        h2, c2 = self.lstm2(h1,window_view(x_weather,self.window_size))  # ([batch_size, seq_size, hidden_size], [batch_size, seq_size, hidden_size])
        TimeProfiler.end("LSTM L2 Forward")

        TimeProfiler.begin("LSTM Out1 Forward")
        out = h2.contiguous().view(-1, self.hidden_dim)  # out size: (24 * batch_size, hidden_dim)
        TimeProfiler.end("LSTM Out1 Forward")

        TimeProfiler.begin("LSTM Out2 Forward")
        out = self.fc(out)  # (24 * batch_size)
        TimeProfiler.end("LSTM Out2 Forward")

        TimeProfiler.begin("LSTM Out3 Forward")
        res = (out).view_as(h2[...,0])
        TimeProfiler.end("LSTM Out3 Forward")

        return res
            # torch.sigmoid(out).view_as(h2[...,0])