from models2.WKAN import WKAN
import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, BatchNorm, global_mean_pool
import pywt
import numpy as np


class GWAN(torch.nn.Module):
    def __init__(self,
                 feature,
                 out_channel,
                 skip: bool = True,
                 hidden_layers: int = 1,
                 dropout: float = 0.):
        super().__init__()
        #------------------------------------------------------------
        mp_layers = 2
        num_features = 512
        hidden_channels = 512
        num_classes = out_channel
        #-----------------------------------------------------------

        self.WavAtt = WaveletAttention()
        self.convs = torch.nn.ModuleList()
        for i in range(mp_layers - 1):
            if i == 0:
                self.convs.append(GWAConvLayer(num_features, hidden_channels, hidden_channels, hidden_layers))
            else:
                self.convs.append(GWAConvLayer(hidden_channels, hidden_channels, hidden_channels, hidden_layers))

        self.skip = skip

        dim_out_message_passing = 512*3 # 1536  for hidden dimensions  512, 1280 for 1024
        self.dropout = torch.nn.Dropout(dropout)

        self.bn1 = BatchNorm(hidden_channels)
        self.bn2 = BatchNorm(dim_out_message_passing)

        self.fc = nn.Sequential(nn.Linear(dim_out_message_passing, 512),
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.2),
                                nn.Linear(512, out_channel)
                                )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        l = []
        l.append(x)
        x = self.WavAtt(x)

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = self.bn1(x)
            x = self.dropout(x)
            l.append(x)

        if self.skip:
            x = torch.cat(l, dim=1)
        x = self.bn2(x)
        x = global_mean_pool(x, batch)

        x = self.fc(x)

        return x


class WaveletAttention(nn.Module):
    def __init__(self, wavelet='db1'):
        super(WaveletAttention, self).__init__()
        self.wavelet = wavelet
        self.attention_dense = nn.Linear(2, 1, bias=False)  # 两个小波分量

    def wavelet_decompose(self, signal):
        signal_np = signal.cpu().numpy()

        low_freq, high_freq = zip(*[pywt.dwt(sig, self.wavelet) for sig in signal_np])

        low_freq = torch.tensor(np.array(low_freq), dtype=signal.dtype, device=signal.device)
        high_freq = torch.tensor(np.array(high_freq), dtype=signal.dtype, device=signal.device)

        return low_freq, high_freq

    def forward(self, x):
        low_freq, high_freq = self.wavelet_decompose(x)

        low_high = torch.stack([low_freq, high_freq], dim=-1)  # (batch_size, signal_length/2, 2)
        attention_scores = torch.sigmoid(self.attention_dense(low_high)).squeeze(-1)
        attended_signal = attention_scores * low_freq + (1 - attention_scores) * high_freq

        return attended_signal


class GWAConvLayer(GINConv):
    def __init__(self, in_feat:int,
                 out_feat:int,
                 hidden_dim:int=512,
                 nb_layers:int=1):
        wkan  =make_wkan(in_feat, hidden_dim, out_feat, nb_layers)
        GINConv.__init__(self, wkan)

def make_wkan(num_features, hidden_dim, out_dim, hidden_layers):
    sizes = [num_features] + [hidden_dim]*(hidden_layers-1) + [out_dim]

    return(WKAN(layers_hidden=sizes,wavelet_type='mexican_hat'))

