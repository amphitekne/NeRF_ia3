from dataclasses import dataclass
from .camera import Camera
from .utils import load_from_colmap_text, get_roi
import random
import os
import torch


def split_by_size(cameras, test_size: float = 0.2) -> tuple:
    num_cameras = len(cameras)
    index = list(range(num_cameras))
    test_index = sorted(random.sample(index, int(num_cameras * test_size)))
    train_index = [i for i in index if i not in test_index]
    train_cameras = [cameras[camera_index] for camera_index in train_index]
    test_cameras = [cameras[camera_index] for camera_index in test_index]

    return (SceneData(cameras=train_cameras, is_train_data=True),
            SceneData(cameras=test_cameras, is_train_data=False))


def split_from_files(scene_path: str, cameras: list) -> tuple:
    with open(os.path.join(scene_path, "gaussian_model", "train_images.txt"), "r") as file:
        train_images_names = [line.replace("\n", "") for line in file]
    with open(os.path.join(scene_path, "gaussian_model", "test_images.txt"), "r") as file:
        test_images_names = [line.replace("\n", "") for line in file]
    train_cameras = []
    test_cameras = []
    for camera in cameras:
        if camera.image_path.split("\\")[-1] in train_images_names:
            train_cameras.append(camera)
        elif camera.image_path.split("\\")[-1] in test_images_names:
            test_cameras.append(camera)
    return (SceneData(cameras=train_cameras, path=scene_path, is_train_data=True),
            SceneData(cameras=test_cameras, path=scene_path, is_train_data=False))


@dataclass
class SceneData:
    cameras: list[Camera]
    path: str
    is_train_data: bool = True

    def __post_init__(self):
        if self.is_train_data:
            self.roi = get_roi(self.cameras)

    def __len__(self):
        return len(self.cameras)

    def get_camera(self, idx: int) -> Camera:
        return self.cameras[idx]

    @property
    def image_width(self) -> int:
        return self.cameras[0].width

    @property
    def image_height(self) -> int:
        return self.cameras[0].height

    def split(self, test_size: None | float = 0.2) -> tuple:
        if test_size is None:
            return split_from_files(scene_path=self.path, cameras=self.cameras)
        else:
            return split_by_size(cameras=self.cameras, test_size=test_size)

    def __getitem__(self, batch_size: int) -> torch.Tensor:
        rays = [self.cameras[random.randint(0, len(self.cameras) - 1)].get_random_ray() for _ in range(batch_size)]
        return torch.stack(rays).squeeze(1)


def get_scene(scene_path: str) -> SceneData:
    cameras = load_from_colmap_text(scene_path=scene_path)
    return SceneData(cameras=cameras, path=scene_path)


def get_scene_split(scene_path: str, test_size: None) -> tuple[SceneData, SceneData]:
    scene = get_scene(scene_path)
    scene_train, scene_test = scene.split(test_size)
    return scene_train, scene_test
