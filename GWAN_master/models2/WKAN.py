
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class WKANLinear(nn.Module):
    def __init__(self, in_features, out_features, wavelet_type='mexican_hat'):
        super(WKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type

        # Parameters for wavelet transformation
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.translation = nn.Parameter(torch.tensor(0.0))
        self.weight1 = nn.Parameter(torch.Tensor(out_features,
                                                 in_features))  # not used; you may like to use it for wieghting base activation and adding it like Spl-KAN paper
        self.wavelet_weights = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))

        # Base activation function #not used for this experiment
        self.base_activation = nn.SiLU()

        # Batch normalization
        self.bn = nn.BatchNorm1d(out_features)

    def wavelet_transform(self, x):

        self.translation_expanded = self.translation.expand(x.size(0))
        self.scale_expanded = self.scale.expand(x.size(0))
        x_scaled = (x - self.translation_expanded[:,None]) / self.scale_expanded[:,None]
        # Implementation of different wavelet types
        if self.wavelet_type == 'mexican_hat':
            # term1 = ((x_scaled ** 2) - 1)
            term1 = (1-(x_scaled ** 2))
            term2 = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = (2 / (math.sqrt(3) * math.pi ** 0.25)) * term1 * term2
            wavelet_output = wavelet

        elif self.wavelet_type == 'morlet':
            omega0 = 5.0  # Central frequency
            real = torch.cos(omega0 * x_scaled)
            envelope = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = envelope * real
            wavelet_output = wavelet

        elif self.wavelet_type == 'laplace':
            A = 0.08
            ep = 0.03
            w = 2 * math.pi * 50
            indc = -ep / (torch.sqrt(torch.tensor(1 - pow(ep, 2))))
            wavelet = A * torch.exp(indc * w * x_scaled) * torch.sin(w * x_scaled)
            wavelet_output = wavelet
        elif self.wavelet_type == 'dog':
            # Implementing Derivative of Gaussian Wavelet
            dog = -x_scaled * torch.exp(-0.5 * x_scaled ** 2)
            wavelet = dog
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)

        elif self.wavelet_type == 'meyer':
            # Implement Meyer Wavelet here
            # Constants for the Meyer wavelet transition boundaries
            v = torch.abs(x_scaled)
            pi = math.pi

            def meyer_aux(v):
                return torch.where(v <= 1 / 2, torch.ones_like(v),
                                   torch.where(v >= 1, torch.zeros_like(v), torch.cos(pi / 2 * nu(2 * v - 1))))

            def nu(t):
                return t ** 4 * (35 - 84 * t + 70 * t ** 2 - 20 * t ** 3)

            # Meyer wavelet calculation using the auxiliary function
            wavelet = torch.sin(pi * v) * meyer_aux(v)
            wavelet_output = wavelet
        elif self.wavelet_type == 'shannon':
            # Windowing the sinc function to limit its support
            pi = math.pi
            sinc = torch.sinc(x_scaled / pi)  # sinc(x) = sin(pi*x) / (pi*x)

            # Applying a Hamming window to limit the infinite support of the sinc function
            window = torch.hamming_window(x_scaled.size(-1), periodic=False, dtype=x_scaled.dtype,
                                          device=x_scaled.device)
            # Shannon wavelet is the product of the sinc function and the window
            wavelet = sinc * window
            wavelet_output = wavelet
            # You can try many more wavelet types ...
        else:
            raise ValueError("Unsupported wavelet type")

        return wavelet_output, self.scale, self.translation

    def forward(self, x):
        wavelet_output, s, u = self.wavelet_transform(x)
        base_output = F.linear(x, self.weight1)
        combined_output = wavelet_output  # + base_output

        return combined_output, s, u


class WKAN(nn.Module):
    def __init__(self, layers_hidden, wavelet_type='mexican_hat'):
        super(WKAN, self).__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(WKANLinear(in_features, out_features, wavelet_type))



    def forward(self, x):
        for layer in self.layers:
            x, s, u = layer(x)

        return x
