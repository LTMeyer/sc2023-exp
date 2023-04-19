from typing import Tuple
import os
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from offline_tools import TensorboardLogger, Net, checkpoint
from process_dataset import NumpyDataset

"""
Script used for running an offline comparison of the heat-pde example.

The study should be executed by navigating to `examples/heat-pde/heat-pde-dl/offline-example`
and running:

python3 run_offline_study.py

The script creates a folder named `offline-<YEAR><MONTH><DAY><TIME>`, then
generates data for n_simulations parameter combinations of heat-pde
solutions. These solutions are stored in a folder called `Res`. Next,
a neural network is trained on these solutions using `batch_size` and `epochs`.
The final trained model is saved as `model.ckpt` and is stored in
`offline-<YEAR><MONTH><DAY><TIME>` which can be used for post processing.

The script depends on the user having already built `heat_no_melissac`, which can
be done by navigating to `examples/heat-pde/executables/build` and executing the
following:

cmake ..
make
"""


def setup_environment_slurm():
    """
    Uses JZ recommendations for setting up DDP environment with slurm
    """
    from melissa.utility import idr_torch

    if torch.cuda.is_available():
        torch.cuda.set_device(idr_torch.local_rank)
        world_size = idr_torch.size
        device = f"cuda:{idr_torch.local_rank}"
        idr_rank = idr_torch.local_rank
        print(f"World size {world_size} rank {idr_rank}")
        backend = "nccl"
    else:
        raise RuntimeError

    dist.init_process_group(
        backend, init_method="env://", rank=idr_torch.rank, world_size=idr_torch.size
    )
    return device, idr_rank, world_size


def get_on_disk_dataloader(
    folder: str,
    train_size: int,
    n_valid_size: int,
    batch_size: int,
    num_workers: int,
    mesh_size: Tuple[int, int],
    num_replica: int,
    rank: int,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_data_folder = os.path.join(folder, "sc2023-heatpde-training")
    valid_data_folder = os.path.join(folder, "sc2023-heatpde-validation")
    train_dataset = NumpyDataset(train_data_folder, train_size, 100, mesh_size)

    train_sampler: DistributedSampler = DistributedSampler(
        train_dataset, num_replicas=num_replica, rank=rank
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=num_workers,
        sampler=train_sampler,
    )

    valid_dataset = NumpyDataset(valid_data_folder, n_valid_size, 100, mesh_size)
    valid_sampler: DistributedSampler = DistributedSampler(
        valid_dataset, num_replicas=num_replica, rank=rank, shuffle=False
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, num_workers=num_workers, sampler=valid_sampler
    )

    return train_dataloader, valid_dataloader


def train_model(args, logger, train_dataloader, valid_dataloader, device, idr_rank, frequency):

    frequency = frequency
    mesh_size = args.mesh_size
    model = Net(6, mesh_size * mesh_size, 1).to(device)
    model = DDP(model, device_ids=[idr_rank])
    model.train()

    # Set Model exactly the same as heatpde_server.py
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    milestones = [int(step) for step in range(6250, 31250, 6250)]
    learning_rate_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.5
    )

    print(f"Validation dataset size {len(valid_dataloader.dataset)}")
    print("Starting training loop")
    last_batch_time = time.time()
    # Train loop exactly the same as heatpde_server.py
    total_batches = 0
    for epoch in range(0, args.epochs):
        print(f"On epoch {epoch + 1} of {args.epochs}")
        start = time.time()
        for batch, batch_data in enumerate(train_dataloader):
            x, y_target = batch_data
            x = x.float().to(device)  # type: ignore
            y_target = y_target.float().to(device)
            y_pred = model(x).view(-1, mesh_size, mesh_size)
            loss = criterion(y_pred, y_target)
            # Backprogation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_batches += 1

            logger.log_scalar("Loss/train", loss.item(), total_batches)
            if total_batches > 0 and total_batches % frequency == 0:
                print(f"On batch {batch + 1}")
                # compute and log validation loss
                samples = args.batch_size * frequency
                end_time = time.time()
                samples_per_second = samples / (end_time - last_batch_time)
                logger.log_scalar("samples_per_second", samples_per_second, batch)
                val_loss = torch.Tensor([0.0]).to(device)
                model.eval()
                valid_batch = 0
                with torch.no_grad():
                    for valid_batch, batch_data in enumerate(valid_dataloader):
                        x, y_target = batch_data
                        x = x.float().to(device)  # type: ignore
                        y_target = y_target.float().to(device)
                        # model evaluation
                        y_pred = model(x).view(-1, mesh_size, mesh_size)
                        loss = criterion(y_pred, y_target)
                        val_loss += loss.item() * y_target.size(0)

                dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
                val_loss = val_loss.cpu() / len(valid_dataloader.dataset)

                last_batch_time = time.time()
                model.train()
                logger.log_scalar("Loss/valid", val_loss, total_batches)

            # Step learning rate
            if args.lr == "stepped":
                learning_rate_scheduler.step()
            lrs = []
            for grp in optimizer.param_groups:
                lrs.append(grp["lr"])
            logger.log_scalar("lr", lrs[0], total_batches)

        time_spent = time.time() - start
        print(f"Epoch time {time_spent:.1f}")

    if idr_rank == 0:
        checkpoint(
            model=model,
            optimizer=optimizer,
            batch=batch,
            loss=loss.item(),
            path=f"{args.out_dir}/tb_{args.id}/model.ckpt",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch-size", type=int, default=10, help="batch size for train loop.")
    parser.add_argument(
        "--num_workers", type=int, default=1, help="number of workers for data loading."
    )
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs to train on.")
    parser.add_argument("--train", type=bool, default=False, help="to train or not")
    parser.add_argument(
        "--ntrain_sims", type=int, default=250, help="number of training solutions"
    )
    parser.add_argument(
        "--nval_sims", type=int, default=10, help="number of validation solutions"
    )
    parser.add_argument("--lr", type=str, default="stepped", help="how to update lr.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=datetime.now().strftime("melissa-%Y%m%dT%H%M%S"),
        help="Where output data are written",
    )
    parser.add_argument(
        "--id", type=str, default=np.random.randint(int(1e6)), help="number of recorded steps"
    )
    parser.add_argument("--mesh_size", type=int, default=100, help="edge mesh discretization")
    parser.add_argument("--data_dir", type=str, help="Path to data folder.", default=None)
    parser.add_argument("--frequency", type=int, default=10, help="validation frequency")

    args = parser.parse_args()
    print(args)

    try:
        device, idr_rank, world_size = setup_environment_slurm()
    except Exception:
        # check for cuda and use if possible
        if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
            device = torch.device("cuda:0")
            print("Found cuda enabled gpu")
            backend = "nccl"
        else:
            device = torch.device("cpu:0")
            print("Using CPU")
            backend = "gloo"
        # set up sequential distributed training
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        world_size = 1
        idr_rank = 0
        dist.init_process_group(
            backend, init_method="env://", rank=idr_rank, world_size=world_size
        )

    os.makedirs(args.out_dir, exist_ok=True)
    logger = TensorboardLogger(rank=0, logdir=f"{args.out_dir}/tb_{args.id}")

    data_dir = args.data_dir
    train_dataloader, valid_dataloader = get_on_disk_dataloader(
        data_dir,
        args.ntrain_sims,
        args.nval_sims,
        args.batch_size,
        args.num_workers,
        (args.mesh_size, args.mesh_size),
        world_size,
        idr_rank,
    )
    if args.train:
        train_model(
            args, logger, train_dataloader, valid_dataloader, device, idr_rank, args.frequency
        )
