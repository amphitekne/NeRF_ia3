import numpy as np
import tinycudann as tcnn
import torch
import torch.nn as nn

from models.utils.activation import trunc_exp


class NeRF(nn.Module):
    def __init__(self, hidden_dim_sigma=64, num_layers_sigma=2, hidden_dim_rgb=64, num_layers_rgb=3,
                 roi: list[float] = [-8.0, -8.0, -8.0, 8.0, 8.0, 8.0]):
        super(NeRF, self).__init__()

        geo_feat_dim = 15

        self.hidden_dim_sigma = hidden_dim_sigma
        self.num_layers_sigma = num_layers_sigma
        self.hidden_dim_rgb = hidden_dim_rgb
        self.num_layers_rgb = num_layers_rgb

        self.geo_feat_dim = geo_feat_dim
        self.roi = max([abs(x) for x in roi])

        per_level_scale = np.exp2(np.log2(2048 * self.roi) / (16 - 1))

        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": per_level_scale,
            },
        )

        self.sigma_net = tcnn.Network(
            n_input_dims=self.encoder.n_output_dims,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "CutlassMLP" if self.hidden_dim_sigma > 128 else "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.hidden_dim_sigma,
                "n_hidden_layers": self.num_layers_sigma - 1,
            },
        )

        # color network

        self.encoder_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.in_dim_color = self.encoder_dir.n_output_dims + self.geo_feat_dim

        self.color_net = tcnn.Network(
            n_input_dims=self.in_dim_color,
            n_output_dims=3,
            network_config={
                "otype": "CutlassMLP" if self.hidden_dim_rgb > 128 else "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.hidden_dim_rgb,
                "n_hidden_layers": self.num_layers_rgb - 1,
            },
        )

    def density(self, x):
        # x: [N, 3], in [-roi, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        x = (x + self.roi) / (2 * self.roi)  # to [0, 1]
        x = self.encoder(x)

        h = self.sigma_net(x)

        sigma = trunc_exp(h[..., 0])

        return sigma

    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], normalized in [-1, 1]

        # sigma
        x = (x + self.roi) / (2 * self.roi)  # to [0, 1]
        x = self.encoder(x)

        h = self.sigma_net(x)

        sigma = trunc_exp(h[..., 0])

        geo_feat = h[..., 1:]

        # color
        d = (d + 1) / 2  # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.encoder_dir(d)

        # p = torch.zeros_like(geo_feat[..., :1]) # manual input padding
        h = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h)

        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return color, sigma
