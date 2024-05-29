import json
import os
from datetime import datetime

import torch
from torch.cuda.amp.grad_scaler import GradScaler
from tqdm import tqdm

from render.rendering import render_rays
from .utils.report import update_report, start_report
from .utils.model_save import save_model


def compute_loss(image: torch.Tensor, gt_image: torch):
    return ((image - gt_image) ** 2).mean()


def test(model, estimator, test_steps, dataloader, render_step_size, tn, tf, device) -> float:
    model.eval()
    estimator.eval()
    total_loss = 0
    with torch.no_grad():
        for _ in tqdm(range(test_steps)):
            batch = dataloader.get_batch()
            o = batch[:, :3].to(device)
            d = batch[:, 3:6].to(device)
            target = batch[:, 6:].to(device)
            prediction, _, _, _, _ = render_rays(model, estimator, render_step_size, o, d, tn, tf, device=device)
            loss = compute_loss(prediction, target)
            total_loss += loss.item()
        avg_test_loss = total_loss / test_steps
    return avg_test_loss


def training(model,
             estimator,
             optimizer,
             scheduler,
             tn,
             tf,
             render_step_size,
             num_epochs,
             num_steps,
             train_dataloader,
             test_dataloader,
             device="cuda",
             project_path: str = ""):
    scaler = GradScaler()

    def occ_eval_fn(x):
        with torch.no_grad():
            density = model.density(x)
        return density * render_step_size

    training_losses = []
    testing_losses = []

    training_directory = os.path.join(project_path, f"model_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
    os.mkdir(training_directory)
    details = {'tn': tn,
               'tf': tf,
               'render_step_size': render_step_size,
               }

    with open(os.path.join(training_directory, 'training_config.json'), 'w') as convert_file:
        convert_file.write(json.dumps(details))
        datetime.now()
    with open(os.path.join(training_directory, "architecture.txt"), "w") as f:
        f.write(str(model))

    start_report(checkpoint_path=training_directory)
    test_steps = 5
    for epoch in range(num_epochs):
        # update occupancy grid
        total_loss = 0
        model.train(True)
        estimator.train(True)
        for step in tqdm(range(num_steps)):
            batch = train_dataloader.get_batch()
            o = batch[:, :3].to(device)
            d = batch[:, 3:6].to(device)
            target = batch[:, 6:].to(device)

            estimator.update_every_n_steps(
                step=step,
                occ_eval_fn=occ_eval_fn,
                occ_thre=1e-2, )

            optimizer.zero_grad()
            prediction, _, _, _, _ = render_rays(model, estimator, render_step_size, o, d, tn, tf, device=device)
            loss = compute_loss(prediction, target)

            # With GradScaler
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            torch.cuda.empty_cache()

        avg_train_loss = total_loss / num_steps

        training_losses.append(avg_train_loss)
        if avg_train_loss < min(training_losses[:-1], default=False):
            save_model(model, estimator, training_directory)

        # Test model
        avg_test_loss = test(model, estimator, test_steps, test_dataloader, render_step_size, tn, tf, device)
        testing_losses.append(avg_test_loss)

        update_report(
            checkpoint_path=training_directory, iteration=epoch,
            training_loss=avg_train_loss, testing_loss=avg_test_loss,
            training_ssim=0, testing_ssim=0,
            training_psnr=0, testing_psnr=0,
        )

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')
        scheduler.step(avg_train_loss)

    return training_losses, testing_losses, training_directory
