import os
from time import time


def start_report(checkpoint_path: str):
    with open(os.path.join(checkpoint_path, "report.csv"), 'w') as file:
        pass
    with open(os.path.join(checkpoint_path, "report.csv"), 'a') as file:
        file.write(
            "iteration;timestamp;training_loss;testing_loss;training_ssim;testing_ssim;training_psnr;testing_psnr" + '\n')


def update_report(checkpoint_path: str, iteration: int,
                  training_loss: float, testing_loss: float,
                  training_ssim: float, testing_ssim: float,
                  training_psnr: float, testing_psnr: float):
    with open(os.path.join(checkpoint_path, "report.csv"), 'a') as file:
        new_line = f"{iteration};{time()};{training_loss};{testing_loss};{training_ssim};{testing_ssim};{training_psnr};{testing_psnr}"
        file.write(new_line + '\n')
