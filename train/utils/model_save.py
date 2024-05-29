import torch
import os


def save_model(model, estimator, directory):
    device = estimator.device
    torch.save(model.cpu(), os.path.join(directory, f"model"))
    torch.save(model.state_dict(), os.path.join(directory, f"model_weights"))
    model.to(device)
    torch.save(estimator.cpu(), os.path.join(directory, f"estimator"))
    torch.save(estimator.state_dict(), os.path.join(directory, f"estimator_weights"))
    estimator.to(device)
