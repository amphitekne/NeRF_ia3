import torch
from torch import nn
import numpy as np
from random import randint


class Camera(nn.Module):
    device = "cuda"

    def __init__(self, id: str, image_path: str, image: torch.Tensor, fl_x: float, fl_y: float, c_x: float,
                 c_y: float, width: int, height: int, translation_matrix: np.array, rotation_matrix: np.array):
        super(Camera, self).__init__()
        self.id = id
        self.image_path = image_path
        self.image = image.squeeze()
        self.fl_x = fl_x
        self.fl_y = fl_y
        self.c_x = c_x
        self.c_y = c_y
        self.width = width
        self.height = height
        self.translation_matrix = translation_matrix
        self.rotation_matrix = rotation_matrix

    def get_random_ray(self):
        width_index = randint(0, self.width - 1)
        height_index = randint(0, self.height - 1)

        x = torch.ones(1) * (width_index - self.c_x) / self.fl_x
        y = -torch.ones(1) * (height_index - self.c_y) / self.fl_y
        z = -torch.ones(1)

        origins = self.translation_matrix.unsqueeze(0).reshape(-1, 3)
        rotation = self.rotation_matrix.unsqueeze(0)

        xyz = torch.stack([x, y, z], dim=-1)
        dirs = torch.einsum("bj, bij -> bi", xyz, rotation)
        dirs = dirs / torch.norm(dirs, dim=-1)[:, None]

        colors = self.image[height_index, width_index].unsqueeze(0) / 255.0
        return torch.cat((origins, dirs, colors), dim=1)

    def get_all_image_rays(self):
        # indices: (B, (cam_idx, u, v))

        u = torch.arange(self.width)
        v = torch.arange(self.height)

        # Create a meshgrid for the y and x positions
        v, u = torch.meshgrid(v, u, indexing='ij')
        pixel_positions = torch.stack([v, u], dim=-1).reshape(-1, 2)
        v = pixel_positions[:, 0]
        u = pixel_positions[:, 1]

        rotation = self.rotation_matrix.unsqueeze(0)
        origins = (torch.zeros((1, self.width * self.height, 3)) +
                   self.translation_matrix.unsqueeze(0))
        origins = origins.reshape(-1, 3)

        x = (u - self.c_x) / self.fl_x
        y = -(v - self.c_y) / self.fl_y
        z = -torch.ones(len(u))

        xyz = torch.stack([x, y, z], dim=-1)
        dirs = torch.einsum("bj, bij -> bi", xyz, rotation)
        dirs = dirs / torch.norm(dirs, dim=-1)[:, None]

        colors = self.image[v, u] / 255.0

        return torch.cat((origins, dirs, colors), dim=1)
