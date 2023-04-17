
import torch
from typing import Any, Optional
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import random
import glob

"""
Tools used for making an offline comparison of the heat-pde example
"""


class TensorboardLogger:
    def __init__(self, rank: int, logdir: str = "tensorboard"):
        self.writer: Optional[SummaryWriter] = SummaryWriter(logdir) if rank == 0 else None

    def log_scalar(self, tag: str, scalar_value: Any, step: int):
        if self.writer:
            self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        if self.writer:
            self.writer.flush()
            self.writer.close()


class Net(torch.nn.Module):
    def __init__(self, input_features, output_features, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.output_features = output_features
        self.hidden_features = 256
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_features, self.hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_features, self.hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_features,
                            output_features * output_dim),
        )

    def forward(self, x):
        y = self.net(x)
        return y


def checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: int,
    loss: float,
    path: str = "model.ckpt",
):
    torch.save(
        {
            "batch": batch,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def draw_param_set(nb_param: int):
    param_set = []
    for i in range(nb_param):
        param_set.append(random.randint(100, 500))
    return param_set


def get_file_result(mesh_size):
    result_files = sorted(glob.glob("./Res/*.dat"),
                          key=lambda f: int(f.split("_")[1]))
    results = []
    for file in result_files:
        result = np.loadtxt(file)
        result = result[:, 1].reshape(mesh_size, mesh_size)
        results.append(result)

    results = np.array(results)
    return results
