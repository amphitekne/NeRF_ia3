import random
from dataclasses import dataclass, field
from .transform_matrix import get_transform_matrix_list

from PIL import Image

import os
import numpy as np
from pathlib import Path

import torch
from torchvision.io import read_image
from tqdm import tqdm


def load_image(image_path: str) -> torch.Tensor:
    # return read_image(image_path) * -1
    _image = Image.open(image_path)
    _img = np.array(_image)
    return torch.from_numpy(_img[None, ...]).type(torch.uint8)


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


def read_colmap(scene_path: str | Path) -> tuple:
    def get_images_data() -> dict:
        with open(os.path.join(scene_path, "colmap_text", "images.txt"), "r") as file:
            text_lines = [line.split("\n")[0].split(" ") for line in file]
        text_lines = text_lines[4::2]
        images_data = {
            line[0]: {"qvec": np.array(line[1:5]).astype(np.float32), "tvec": np.array(line[5:8]).astype(np.float32),
                      "camera_id": line[8], "image_name": line[9]} for line in text_lines}
        return images_data

    def get_camera_data() -> dict:
        with open(os.path.join(scene_path, "colmap_text", "cameras.txt"), "r") as file:
            text_lines = [line.split("\n")[0].split(" ") for line in file]
        text_lines = text_lines[3:]
        camera_data = [{"model": line[1],
                        "width": int(line[2]),
                        "height": int(line[3]),
                        "fl_x": float(line[4]),
                        "fl_y": float(line[5]),
                        "c_x": float(line[6]),
                        "c_y": float(line[7]),
                        }
                       for line in text_lines]
        return camera_data[0]

    camera_data = get_camera_data()
    images_data = get_images_data()

    return camera_data, images_data


def train_test_split(_frames_path_list, _transform_matrix_list, _n):
    # Ensure n is not greater than the length of the list
    _n = min(_n, len(_frames_path_list))

    # Generate n unique random indices
    _selected_indices = random.sample(range(len(_frames_path_list)), _n)

    _train_data_dict = {
        "frames_path_list": [_frames_path_list[i] for i in range(len(_frames_path_list)) if i not in _selected_indices],
        "transform_matrix_list": [_transform_matrix_list[i] for i in range(len(_transform_matrix_list)) if
                                  i not in _selected_indices]}
    _test_data_dict = {"frames_path_list": [_frames_path_list[i] for i in _selected_indices],
                       "transform_matrix_list": [_transform_matrix_list[i] for i in _selected_indices]}

    return _train_data_dict, _test_data_dict


def train_test_split_from_files(nerf_data, train_images_names, test_images_names):
    frames_path_list = nerf_data.frames_path_list
    transform_matrix_list = nerf_data.transform_matrix_list

    train_data = {
        "frames_path_list": [],
        "transform_matrix_list": []
    }
    for frames_path, transform_matrix in zip(frames_path_list, transform_matrix_list):
        if frames_path.split("\\")[-1] in train_images_names:
            train_data["frames_path_list"].append(frames_path)
            train_data["transform_matrix_list"].append(transform_matrix)
    test_data = {
        "frames_path_list": [],
        "transform_matrix_list": []
    }
    for frames_path, transform_matrix in zip(frames_path_list, transform_matrix_list):
        if frames_path.split("\\")[-1] in test_images_names:
            test_data["frames_path_list"].append(frames_path)
            test_data["transform_matrix_list"].append(transform_matrix)

    return train_data, test_data


class ImagesDataset:
    def __init__(self,
                 _project_path: str,
                 _frames_path_list: list,
                 _transform_matrix_list: list,
                 _intrinsics_matrix: np.array,
                 _image_size: tuple,
                 _device: str = "cuda"):
        self._project_path = _project_path
        self._frames_path_list = _frames_path_list
        self.poses = torch.Tensor(np.array(_transform_matrix_list))
        self._intrinsics_matrix = _intrinsics_matrix
        self._image_size = _image_size
        self._device = _device

        self._n_images = len(self._frames_path_list)
        self._n_rays_per_image = self._image_size[0] * self._image_size[1]

        self.images = self.__load_images()
        self._image_size = (self.images[0].shape[1], self.images[0].shape[0])

    def __len__(self):
        return self._n_images * self._n_rays_per_image

    def __load_image(self, _index: int) -> torch.Tensor:
        _frame_path = self._frames_path_list[_index]
        _image = Image.open(os.path.join(self._project_path, _frame_path.split("./")[-1]))
        _img = np.array(_image)
        return torch.from_numpy(_img[None, ...]).type(torch.uint8)

    def __load_images(self):
        images = []
        for _index in tqdm(range(self._n_images)):
            images.append(self.__load_image(_index))
        return torch.squeeze(torch.Tensor(np.array(images)).type(torch.uint8))

    def get_image_rays(self, _index) -> torch.Tensor:
        # indices: (B, (cam_idx, u, v))
        img_index = _index

        _u = torch.arange(self._image_size[0])  # width
        _v = torch.arange(self._image_size[1])  # height

        # Create a meshgrid for the y and x positions
        _v, _u = torch.meshgrid(_v, _u, indexing='ij')
        pixel_positions = torch.stack([_v, _u], dim=-1).reshape(-1, 2)
        _v = pixel_positions[:, 0]
        _u = pixel_positions[:, 1]

        rotation = self.poses[img_index, :3, :3].unsqueeze(0)
        origins = (torch.zeros((1, self._image_size[0] * self._image_size[1], 3)) +
                   self.poses[img_index, :3, 3].unsqueeze(0))
        origins = origins.reshape(-1, 3)
        f_x = self._intrinsics_matrix[0, 0]
        f_y = self._intrinsics_matrix[1, 1]
        c_x = self._intrinsics_matrix[0, 2]
        c_y = self._intrinsics_matrix[1, 2]

        x = (_u - c_x + 0.5) / f_x
        y = -(_v - c_y + 0.5) / f_y

        z = -torch.ones(len(_u))

        xyz = torch.stack([x, y, z], dim=-1)

        dirs = torch.einsum("bj, bij -> bi", xyz, rotation)

        dirs = dirs / torch.norm(dirs, dim=-1)[:, None]
        colors = self.images[img_index, _v, _u] / 255.0
        return torch.cat((origins, dirs, colors), dim=1)

    def get_rays(self, indices) -> torch.Tensor:
        # indices: (B, (cam_idx, u, v))
        img_indices, uv = indices[:, 0], indices[:, 1:]
        rotation = self.poses[img_indices, :3, :3]
        origins = self.poses[img_indices, :3, 3]

        f_x = self._intrinsics_matrix[0, 0]
        f_y = self._intrinsics_matrix[1, 1]
        c_x = self._intrinsics_matrix[0, 2]
        c_y = self._intrinsics_matrix[1, 2]

        # x = (uv[:, 0] - c_x + 0.5) / f_x
        # y = -(uv[:, 1] - c_y + 0.5) / f_y
        x = (uv[:, 0] - c_x + 0.5) / f_x
        y = -(uv[:, 1] - c_y + 0.5) / f_y

        z = -torch.ones(len(uv))
        xyz = torch.stack([x, y, z], dim=-1)

        dirs = torch.einsum("bj, bij -> bi", xyz, rotation)

        dirs = dirs / torch.norm(dirs, dim=-1)[:, None]
        colors = self.images[img_indices, uv[:, 1], uv[:, 0]] / 255.0
        return torch.cat((origins, dirs, colors), dim=1)

    def __getitem__(self, batch_size: int) -> torch.Tensor:
        indices = torch.floor(
            torch.rand(batch_size, 3) * torch.tensor([self._n_images, self._image_size[0], self._image_size[1]])).long()
        return self.get_rays(indices)


class CustomDataloader:
    def __init__(self, custom_dataset: ImagesDataset, batch_size: int):
        self.dataset = custom_dataset
        self.batch_size = batch_size

    def get_batch(self):
        return self.dataset.__getitem__(batch_size=self.batch_size)

    def get_image(self, index):
        return self.dataset.get_image_rays(index)

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_batch()


@dataclass
class NerfData:
    project_path: str
    n_images: int | None = field(default=None)
    intrinsics_matrix: np.array = field(init=False)
    images_width: int = field(init=False)
    images_height: int = field(init=False)
    frames_path_list: list = field(init=False)
    frames_matrix_list: list = field(init=False)

    def __post_init__(self):
        camera_data, images_data = read_colmap(self.project_path)

        self.intrinsics_matrix = np.array([
            [camera_data["fl_x"], 0, camera_data["c_x"], 0],
            [0, camera_data["fl_y"], camera_data["c_y"], 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        frames = [image_data for _, image_data in images_data.items()]
        if self.n_images is not None:
            self.n_images = min(self.n_images, len(frames))
            frames_index = random.sample(range(len(frames)), self.n_images)
            frames = [frames[i] for i in frames_index]
        self.frames_path_list = [os.path.join("images", frame["image_name"]) for frame in frames]

        frames_ = get_transform_matrix_list(frames)
        self.transform_matrix_list = [frame["transform_matrix"] for frame in frames_]

        self.images_width = int(camera_data["width"])
        self.images_height = int(camera_data["height"])

        x_coords = [transform_matrix[0, -1] for transform_matrix in self.transform_matrix_list]
        y_coords = [transform_matrix[1, -1] for transform_matrix in self.transform_matrix_list]
        z_coords = [transform_matrix[2, -1] for transform_matrix in self.transform_matrix_list]

        # Vertices of the cube
        x_min = min(min(x_coords), 0)
        x_max = max(max(x_coords), 0)
        y_min = min(min(y_coords), 0)
        y_max = max(max(y_coords), 0)
        z_min = min(min(z_coords), 0)
        z_max = max(max(z_coords), 0)
        self.bounds = [x_min, y_min, z_min, x_max, y_max, z_max]


def dataloaders(project_path: str, n_train_images: int | None = None, n_test_images: int | None = None,
                batch_size: int = 1000):
    try:
        with open(os.path.join(project_path, "gaussian_model", "train_images.txt"), "r") as file:
            train_images_names = [line.replace("\n", "") for line in file]
        with open(os.path.join(project_path, "gaussian_model", "test_images.txt"), "r") as file:
            test_images_names = [line.replace("\n", "") for line in file]
        nerf_data = NerfData(project_path, None)
        train_data_dict, test_data_dict = train_test_split_from_files(nerf_data=nerf_data,
                                                                      train_images_names=train_images_names,
                                                                      test_images_names=test_images_names
                                                                      )
    except:
        if n_test_images is None:
            n_test_images = 0
        nerf_data = NerfData(project_path, n_train_images)

        train_data_dict, test_data_dict = train_test_split(nerf_data.frames_path_list,
                                                           nerf_data.transform_matrix_list,
                                                           n_test_images)
    print("Loading training data...")
    train_dataset = ImagesDataset(_project_path=project_path,
                                  _frames_path_list=train_data_dict["frames_path_list"],
                                  _transform_matrix_list=train_data_dict["transform_matrix_list"],
                                  _intrinsics_matrix=nerf_data.intrinsics_matrix,
                                  _image_size=(nerf_data.images_width, nerf_data.images_height), )
    train_dataloader = CustomDataloader(train_dataset, batch_size=batch_size)


    print("Loading test data...")
    test_dataset = ImagesDataset(_project_path=project_path,
                                 _frames_path_list=test_data_dict["frames_path_list"],
                                 _transform_matrix_list=test_data_dict["transform_matrix_list"],
                                 _intrinsics_matrix=nerf_data.intrinsics_matrix,
                                 _image_size=(nerf_data.images_width, nerf_data.images_height), )
    test_dataloader = CustomDataloader(test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader, nerf_data.bounds
