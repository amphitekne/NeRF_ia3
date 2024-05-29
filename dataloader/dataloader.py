
from .scene import get_scene, SceneData


class CustomDataloader:
    def __init__(self, scene: SceneData, batch_size: int):
        self.scene = scene
        self.batch_size = batch_size

    def get_batch(self):
        return self.scene.__getitem__(batch_size=self.batch_size)

    def get_image_rays(self, image_index: int):
        return self.scene.cameras[image_index].get_all_image_rays()

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_batch()


def get_dataloaders(scene_path: str):
    scene = get_scene(scene_path)
    scene_train, scene_test = scene.split()

    return scene_train, scene_test
