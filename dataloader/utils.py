import os
import numpy as np
from pathlib import Path
import torch
from PIL import Image
from tqdm import tqdm
from .camera import Camera
from .transform_matrix import get_transform_matrix_list


def load_image(image_path: str) -> torch.Tensor:
    image = Image.open(image_path)
    img = np.array(image)
    return torch.from_numpy(img[None, ...]).type(torch.uint8)


def load_from_colmap_text(scene_path: str | Path) -> list[Camera]:
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
        camera_data = {line[0]: {"model": line[1],
                                 "width": int(line[2]),
                                 "height": int(line[3]),
                                 "fl_x": float(line[4]),
                                 "fl_y": float(line[5]),
                                 "c_x": float(line[6]),
                                 "c_y": float(line[7]),
                                 }
                       for line in text_lines}
        return camera_data

    camera_data = get_camera_data()
    images_data = get_images_data()
    frames = [image_data for _, image_data in images_data.items()]
    frames = get_transform_matrix_list(frames)
    transform_matrix_list = torch.Tensor(np.array([frame["transform_matrix"] for frame in frames]))

    cameras = []
    for image_id, data in tqdm(images_data.items()):
        camera_id = data["camera_id"]
        image_path = os.path.join(scene_path, "images", data["image_name"])
        image = load_image(image_path)
        image_index = len(cameras)
        camera = Camera(id=camera_id,
                        image_path=image_path,
                        image=image,
                        width=image.shape[2],
                        height=image.shape[1],
                        fl_x=camera_data[camera_id]["fl_x"],
                        fl_y=camera_data[camera_id]["fl_y"],
                        c_x=camera_data[camera_id]["c_x"],
                        c_y=camera_data[camera_id]["c_y"],
                        rotation_matrix=transform_matrix_list[image_index, :3, :3],
                        translation_matrix=transform_matrix_list[image_index, :3, 3],
                        )
        cameras.append(camera)
    return cameras


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.numpy().transpose()
    Rt[:3, 3] = t.numpy().transpose()
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def get_roi(cameras: list) -> list:
    positions = torch.stack([camera.translation_matrix for camera in cameras])
    min_pos = torch.min(positions, dim=0)[0]
    max_pos = torch.max(positions, dim=0)[0]
    return [min(min_pos[0].item(), 0), min(min_pos[1].item(), 0), min(min_pos[2].item(), 0),
            max(max_pos[0].item(), 0), max(max_pos[1].item(), 0), max(max_pos[2].item(), 0)]


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.rotation_matrix, cam.translation_matrix)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}
