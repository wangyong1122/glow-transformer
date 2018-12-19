import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn

import numpy as np

from glow.layers_glow import *
from glow.utils_glow import *


# ------------------------------------------------------------------------------
# Abstract Classes to define common interface for invertible functions
# ------------------------------------------------------------------------------

# Abstract Class for bijective functions
class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()

    def forward_(self, x, objective):
        raise NotImplementedError

    def reverse_(self, y, objective):
        raise NotImplementedError


# Wrapper for stacking multiple layers
class LayerList(Layer):
    def __init__(self, list_of_layers=None):
        super(LayerList, self).__init__()
        self.layers = nn.ModuleList(list_of_layers)

    def __getitem__(self, i):
        return self.layers[i]

    def forward_(self, x, objective):
        for layer in self.layers:
            x, objective = layer.forward_(x, objective)
        return x, objective

    def reverse_(self, x, objective):
        for layer in reversed(self.layers):
            x, objective = layer.reverse_(x, objective)
        return x, objective

# ActNorm Layer with data-dependant init
class ActNorm(Layer):
    def __init__(self, num_features, logscale_factor=1., scale=1.):
        super(Layer, self).__init__()
        self.initialized = False
        self.logscale_factor = logscale_factor
        self.scale = scale
        self.register_parameter('b', nn.Parameter(torch.zeros(1, num_features, 1)))
        self.register_parameter('logs', nn.Parameter(torch.zeros(1, num_features, 1)))

    def forward_(self, input, objective):

        input_shape = input.size()
        assert len(input_shape)==3

        input = input.view(input_shape[0], input_shape[1], -1)

        if not self.initialized:
            self.initialized = True
            unsqueeze = lambda x: x.unsqueeze(0).unsqueeze(-1).detach()

            # Compute the mean and variance
            sum_size = input.size(0) * input.size(-1)
            input_sum = input.sum(dim=0).sum(dim=-1)
            b = input_sum / sum_size * -1.
            vars = ((input - unsqueeze(b)) ** 2).sum(dim=0).sum(dim=1) / sum_size
            vars = unsqueeze(vars)
            logs = torch.log(self.scale / torch.sqrt(vars) + 1e-6) / self.logscale_factor

            self.b.data.copy_(unsqueeze(b).data)
            self.logs.data.copy_(logs.data)

        logs = self.logs * self.logscale_factor
        b = self.b

        output = (input + b) * torch.exp(logs)
        dlogdet = torch.sum(logs) * input.size(-1)

        return output.view(input_shape), objective + dlogdet

    def reverse_(self, input, objective):
        assert self.initialized
        input_shape = input.size()
        input = input.view(input_shape[0], input_shape[1], -1)
        logs = self.logs * self.logscale_factor
        b = self.b
        output = input * torch.exp(-logs) - b
        dlogdet = torch.sum(logs)* input.size(-1)

        return output.view(input_shape), objective - dlogdet

# Invertible 1x1 convolution
class Invertible1x1Conv(Layer, nn.Conv1d):
    def __init__(self, num_channels):
        self.num_channels = num_channels
        nn.Conv1d.__init__(self, num_channels, num_channels, 1, bias=False)

    def reset_parameters(self):
        # initialization done with rotation matrix
        w_init = np.linalg.qr(np.random.randn(self.num_channels, self.num_channels))[0]
        w_init = torch.from_numpy(w_init.astype('float32'))
        w_init = w_init.unsqueeze(-1)
        self.weight.data.copy_(w_init)

    def forward_(self, x, objective):
        dlogdet = torch.det(self.weight.squeeze()).abs().log()* x.size(-1)
        objective += dlogdet
        output = F.conv1d(x, self.weight, self.bias, self.stride, self.padding, \
                          self.dilation, self.groups)

        return output, objective

    def reverse_(self, x, objective):
        dlogdet = torch.det(self.weight.squeeze()).abs().log()* x.size(-1)
        objective -= dlogdet
        weight_inv = torch.inverse(self.weight.squeeze()).unsqueeze(-1)
        output = F.conv1d(x, weight_inv, self.bias, self.stride, self.padding, \
                          self.dilation, self.groups)
        return output, objective


# Additive Coupling Layer
class AffineCoupling(Layer):
    def __init__(self, num_features, args):
        super(AffineCoupling, self).__init__()
        # assert num_features % 2 == 0
        self.NN = NN(num_features // 2, channels_out=num_features)

    def forward_(self, x, objective):
        z1, z2 = torch.chunk(x, 2, dim=1)
        h = self.NN(z1)
        shift = h[:, 0::2]
        scale = torch.sigmoid(h[:, 1::2] + 2.)
        z2 += shift
        z2 *= scale

        objective += flatten_sum(torch.log(scale))

        return torch.cat([z1, z2], dim=1), objective

    def reverse_(self, x, objective):
        z1, z2 = torch.chunk(x, 2, dim=1)
        h = self.NN(z1)
        shift = h[:, 0::2]
        scale = torch.sigmoid(h[:, 1::2] + 2.)
        z2 /= scale
        z2 -= shift
        objective -= flatten_sum(torch.log(scale))
        return torch.cat([z1, z2], dim=1), objective


# Gaussian Prior that's compatible with the Layer framework
class GaussianPrior(Layer):
    def __init__(self, num_dim, args):
        super(GaussianPrior, self).__init__()
        if args.learntop:
            self.conv = Conv1dZeroInit(2 * num_dim, 2 * num_dim, 3, padding=(3 - 1) // 2)
        else:
            self.conv = None

        self.num_dim = num_dim
        self.args = args

    def forward_(self, x, objective):
        mean_and_logsd = torch.cat([torch.zeros_like(x) for _ in range(2)], dim=1)

        if self.conv:
            mean_and_logsd = self.conv(mean_and_logsd)

        mean, logsd = torch.chunk(mean_and_logsd, 2, dim=1)

        pz = gaussian_diag(mean, logsd)
        objective += pz.logp(x)

        # this way, you can encode and decode back the same image.
        return x, objective

    def reverse_(self, x, objective):
        assert x is None
        bs, num_dim, h = self.args.sample_bs, self.num_dim, 1

        mean_and_logsd = torch.cuda.FloatTensor(bs, 2 * num_dim, h).fill_(0.)

        if self.conv:
            mean_and_logsd = self.conv(mean_and_logsd)

        mean, logsd = torch.chunk(mean_and_logsd, 2, dim=1)
        pz = gaussian_diag(mean, logsd)
        z = pz.sample()
        objective -= pz.logp(z)

        return z, objective


# 1 step of the flow (see Figure 2 a) in the original paper)
class RevNetStep(LayerList):
    def __init__(self, num_channels, args):
        super(RevNetStep, self).__init__()
        self.args = args
        layers = []
        if args.norm == 'actnorm':
            layers += [ActNorm(num_channels)]
        else:
            assert not args.norm

        if args.permutation == 'conv':
            layers += [Invertible1x1Conv(num_channels)]
        else:
            raise ValueError

        if args.coupling == 'affine':
            layers += [AffineCoupling(num_channels, args)]
        else:
            raise ValueError

        self.layers = nn.ModuleList(layers)


# Full model
class GlowMt(LayerList, nn.Module):
    def __init__(self, num_dim, args):
        super(GlowMt, self).__init__()
        layers = []
        output_shapes = []

        layers += [RevNetStep(num_dim, args) for _ in range(args.depth)]

        layers += [GaussianPrior(num_dim, args)]

        self.layers = nn.ModuleList(layers)
        self.output_shapes = output_shapes
        self.args = args
        self.flatten()

    def forward(self, inputs, objective):
        return self.forward_(inputs, objective)

    def sample(self):
        with torch.no_grad():
            samples = self.reverse_(None, 0.)[0]
            return samples

    def flatten(self):
        # flattens the list of layers to avoid recursive call every time.
        processed_layers = []
        to_be_processed = [self]
        while len(to_be_processed) > 0:
            current = to_be_processed.pop(0)
            if isinstance(current, LayerList):
                to_be_processed = [x for x in current.layers] + to_be_processed
            elif isinstance(current, Layer):
                processed_layers += [current]

        self.layers = nn.ModuleList(processed_layers)