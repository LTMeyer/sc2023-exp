import logging
import random
from typing import Dict, Any
import numpy as np
import torch
import torch.utils.data
import time
import os.path as osp

from melissa.server.deep_learning.torch_server import TorchServer, checkpoint
from melissa.server.deep_learning.dataset import MelissaIterableDataset
from melissa.server.deep_learning.reservoir import (
    FIFO, FIRO, Reservoir)
from melissa.server.simulation import SimulationData
from melissa.launcher import message

# from offline.process_dataset import NumpyDataset as OfflineDataset


logger = logging.getLogger("melissa")


class HeatPDEServerDL(TorchServer):
    """
    Use-case specific server
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.param_list = ["ic", "b1", "b2", "b3", "b4", "t"]
        self.mesh_size = self.study_options["mesh_size"]
        random.seed(self.study_options["seed"])

    def set_model(self):
        self.model = self.MyModel(self.nb_parameters + 1, self.mesh_size * self.mesh_size, 1).to(
            self.device
        )

    def configure_data_collection(self):
        """
        function designed to instantiate the data collector and
        buffer
        """

        """self.buffer = FIRO(
            self.buffer_size, self.per_server_watermark, self.pseudo_epochs
        )"""
        self.buffer = self.set_buffer(self.dl_config.get("buffer", "SimpleQueue"))
        self.dataset = MelissaIterableDataset(
            buffer=self.buffer,
            tb_logger=self.tb_logger,
            config=self.config,
            transform=self.process_simulation_data
        )

    def train(self, model: torch.nn.Module):

        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, drop_last=True, num_workers=0
        )

        # Set Model
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.dl_config.get("lr", 1e-3), weight_decay=1e-4
        )

        milestones = [int(step) for step in range(6250, 31250, 6250)]
        learning_rate_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.5
        )

        valid_data_path = self.dl_config.get("valid_data_path", None)
        if valid_data_path:
            n_simulation = self.dl_config.get("n_valid_simulations", 0)
            n_steps = self.study_options.get("num_samples", 0)
            dataloader_valid = self.load_validation_data(valid_data_path, n_simulation, n_steps)
            logger.info("Found valid dataset path for loss validation computation")
        else:
            dataloader_valid = None
            logger.warning("Did not find valid dataset path for loss validation computation")

        logger.info("Start Training")
        last_batch_time = time.time()
        for batch, batch_data in enumerate(dataloader):
            if self.other_processes_finished(batch):
                logger.info("At least one other process has finished. Break training loop.")
                # user adds this to enforce all server procs
                # will stop together and avoid a deadlock
                # in the gradient averaging
                break

            self.tb_logger.log_scalar("put_get_inc", self.buffer.put_get_metric.val, batch)
            if self.dl_config["get_buffer_statistics"]:
                self.get_buffer_statistics(batch)

            # Backprogation
            optimizer.zero_grad()
            x, y_target = batch_data
            x = x.to(self.device)
            y_target = y_target.to(self.device)
            y_pred = model(x)
            loss = criterion(y_pred, y_target)
            loss.backward()
            optimizer.step()

            learning_rate_scheduler.step()

            self.tb_logger.log_scalar("Loss/train", loss.item(), batch)
            if batch > 0 and (batch + 1) % self.n_batches_update == 0:
                samples = self.batch_size * self.n_batches_update * self.num_server_proc
                end_time = time.time()  # get throughput w/out validation
                if dataloader_valid is not None and self.rank == 0:
                    val_loss = 0
                    valid_batch = 0
                    with self.buffer.mutex:  # dont let anything put to the buffer during valid.
                        with torch.no_grad():
                            model.eval()
                            for valid_batch, batch_data in enumerate(dataloader_valid):
                                x, y_target = batch_data
                                x = x.to(self.device)  # type: ignore
                                y_target = y_target.to(self.device)
                                # model evaluation
                                y_pred = model(x)
                                loss = criterion(y_pred, y_target)
                                val_loss += loss.item()
                                del batch_data

                                snd_msg = self.encode_msg(message.Ping())
                                self.launcherfd.send(snd_msg)

                        self.tb_logger.log_scalar("Loss/valid", val_loss / valid_batch, batch)
                        model.train()

                # Learning rate
                lrs = []
                for grp in optimizer.param_groups:
                    lrs.append(grp["lr"])
                    self.tb_logger.log_scalar("lr", lrs[0], batch)
                samples_per_second = samples / (end_time - last_batch_time)
                self.tb_logger.log_scalar("samples_per_second", samples_per_second, batch)
                last_batch_time = time.time()

        logger.info(f"{self.rank} finished training")
        seen_counts = list(self.buffer.seen_ctr.elements())
        if seen_counts:
            self.tb_logger.log_histogram("seen", np.array(seen_counts))
        checkpoint(model=model, optimizer=optimizer, batch=batch, loss=loss.item())

    def draw_parameters(self):
        time_discretization = self.study_options["time_discretization"]
        Tmin, Tmax = self.study_options["parameter_range"]
        param_set = [self.mesh_size, self.mesh_size, time_discretization]
        for i in range(self.study_options["nb_parameters"]):
            param_set.append(random.uniform(Tmin, Tmax))
        return param_set

    def process_simulation_data(cls, msg: SimulationData, config: dict):
        study_options = config["study_options"]
        nb_params = study_options["nb_parameters"]
        # mesh_size = study_options["mesh_size"]

        x = torch.from_numpy(
            np.array(msg.parameters[-nb_params:] + [msg.time_step], dtype=np.float32)
        )
        y = torch.from_numpy(msg.data[0])

        return x, y

    class MyModel(torch.nn.Module):
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
                torch.nn.Linear(self.hidden_features, output_features * output_dim),
            )

        def forward(self, x):
            y = self.net(x)
            return y

    def load_validation_data(self, path: str, n_sim: int, n_steps: int):
        x_valid = np.load(osp.join(path, "input_10.npy"))
        x_valid = x_valid.reshape(-1, 6)
        y_valid = np.load(osp.join(path, "validation_10.npy"))
        y_valid = y_valid.reshape(-1, self.mesh_size * self.mesh_size)
        print(f"Validation data shapes {x_valid.shape}, {y_valid.shape}")
        valid_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(x_valid), torch.from_numpy(y_valid)
        )

        # valid_dataset = OfflineDataset(path, n_sim, n_steps, (self.mesh_size, self.mesh_size))
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=self.batch_size, num_workers=0, shuffle=False
        )

        return valid_dataloader

    def get_buffer_statistics(self, batch: int):
        rows = len(self.buffer.queue)
        columns = int(self.study_options["nb_parameters"]) + 1
        stats_array = np.zeros((rows, columns))
        for i in range(rows):
            item = self.buffer.queue[i]
            x, _ = self.process_simulation_data(item.data, self.config)
            stats_array[i, :] = x

        std, mean = torch.std_mean(torch.tensor(stats_array), dim=0)
        for std_v, mean_v, param in zip(std, mean, self.param_list):
            self.tb_logger.log_scalar(f"buffer_std/{param}", std_v, batch)
            self.tb_logger.log_scalar(f"buffer_mean/{param}", mean_v, batch)

    def set_buffer(self, buffer_str: str):

        if buffer_str == "FIFO":
            buffer = FIFO(self.buffer_size)
        elif buffer_str == "FIRO":
            buffer = FIRO(  # type: ignore
                self.buffer_size,
                self.per_server_watermark,
                self.pseudo_epochs
            )
        elif buffer_str == "Reservoir":
            buffer = Reservoir(  # type: ignore
                self.buffer_size,
                self.per_server_watermark
            )

        return buffer
