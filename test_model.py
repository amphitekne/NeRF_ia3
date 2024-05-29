from test.test import test
import torch
import os
from dataloader.scene import get_scene
import matplotlib.pyplot as plt

# Check if CUDA is available
if torch.cuda.is_available():
    print("Torch CUDA version:", torch.version.cuda)
    print("Torch version:", torch.__version__)

    # Count the number of GPUs
    print(f"Number of GPUs available: {torch.cuda.device_count()}")

    # Get the current GPU index
    current_gpu = torch.cuda.current_device()
    print(f"Current GPU index: {current_gpu}")

    # Get the name of the current GPU
    gpu_name = torch.cuda.get_device_name(current_gpu)
    print(f"Current GPU Name: {gpu_name}")

    device = torch.device("cuda")  # Use GPU
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. No GPU detected.")
    device = torch.device("cpu")  # Use CPU
    print("Using CPU")

scene_path = ""
model_path = ""

model = torch.load(os.path.join(model_path, "model")).to(device)
estimator = torch.load(os.path.join(model_path, "estimator")).to(device)

scene = get_scene(scene_path=scene_path)

camera_index = 1

camera = scene.get_camera(camera_index)

camera_rays = camera.get_all_image_rays()
o = camera_rays[:, :3].to("cuda")
d = camera_rays[:, 3:6].to("cuda")

target = camera_rays[:, 6:].reshape(camera.height, camera.width, 3)

tn = 0.01
roi = scene.roi
extra_bounds = 2
roi = [roi[0] - 1 * extra_bounds,
       roi[1] - 1 * extra_bounds,
       roi[2] - 1 * extra_bounds,
       roi[3] + 1 * extra_bounds,
       roi[4] + 1 * extra_bounds,
       roi[5] + 1 * extra_bounds]
tf = (max(roi) - min(roi)) * 2

prediction = test(model=model.to("cuda"),
                  estimator=estimator.to("cuda"),
                  o=o.to("cuda"), d=d.to("cuda"),
                  H=camera.height, W=camera.width,
                  tn=0.01, tf=tf,
                  render_step_size=0.01,
                  chunk_size=1000, )

prediction = prediction.cpu()
plt.imshow(prediction)
plt.show()
