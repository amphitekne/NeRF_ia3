import nerfacc
import torch

from models.nerf import NeRF
from train.train import training
from dataloader.dataloader_ import dataloaders

# Check if CUDA is available
if torch.cuda.is_available():
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

project_path = ""
batch_size = 1000
n_epochs = 100
hidden_dim_sigma = 64
num_layers_sigma = 2
hidden_dim_rgb = 64
num_layers_rgb = 3
n_steps = 100
resolution = 128
render_step_size = 0.01

print("Loading data...")

train_dataloader, test_dataloader, roi = dataloaders(project_path=project_path,
                                                     n_train_images=None,
                                                     n_test_images=None,
                                                     batch_size=batch_size)

# rays sizing
tn = 0.01
tf = (max(roi) - min(roi)) * 2

extra_bounds = 2
roi = [roi[0] - 1 * extra_bounds,
       roi[1] - 1 * extra_bounds,
       roi[2] - 1 * extra_bounds,
       roi[3] + 1 * extra_bounds,
       roi[4] + 1 * extra_bounds,
       roi[5] + 1 * extra_bounds]

# model definition
print("Building model...")
model = NeRF(hidden_dim_sigma=hidden_dim_sigma, num_layers_sigma=num_layers_sigma,
             hidden_dim_rgb=hidden_dim_rgb, num_layers_rgb=num_layers_rgb,
             roi=roi).to(device)

# Training configuration
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=1)

# Estimator
estimator = nerfacc.OccGridEstimator(roi_aabb=roi, resolution=resolution).to(device)

# Training
print("Training model...")
_, _, training_directory = training(model, estimator, optimizer, scheduler,
                                    tn, tf, render_step_size,
                                    n_epochs, n_steps,
                                    train_dataloader, test_dataloader,
                                    device=device,
                                    project_path=project_path)
