from typing import Tuple
import argparse
import os.path as osp
import time
import re
import glob
from typing import Type, Union

import numpy as np
import numpy.typing as npt
import torch.utils.data


class DatDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, n_sim: int, n_steps: int, mesh_size: Tuple[int, int]):
        self.root = root
        self.n_sim = n_sim
        self.n_steps = n_steps
        self.mesh_size = mesh_size
        self.parameters = self.get_parameters()
        super().__init__()

    def get_parameters(self) -> np.ndarray:
        log_filename = osp.join(self.root, "melissa_server_0.log")
        client_regex = re.compile(r"created client.\d+.sh with parameters")
        parameters_regex = re.compile(r"\[(?P<parameters>.+)\]")
        simulation_parameters = []
        with open(log_filename, "r") as log_file:
            for line in log_file:
                if re.search(client_regex, line):
                    parameter_match = re.search(parameters_regex, line)
                    assert parameter_match
                    parameters_str = parameter_match.group("parameters")
                    parameters = np.fromstring(parameters_str, sep=",", dtype=np.float32)
                    simulation_parameters.append(parameters)
                if len(simulation_parameters) >= self.n_sim:
                    break

        parameters = np.array(simulation_parameters)
        del simulation_parameters
        return parameters

    def __len__(self):
        return self.n_steps * self.n_sim

    def get_indices(self, idx: int):
        time_step_index = idx % self.n_steps
        simulation_index = idx // self.n_steps
        return simulation_index, time_step_index

    def __getitem__(self, idx):
        simulation_index, time_step_index = self.get_indices(idx)
        data_file = osp.join(
            self.root, f"Res_{simulation_index}", f"solution_{time_step_index}.dat"
        )
        data = np.loadtxt(data_file, usecols=1, dtype=np.float32)
        data = data.reshape(*self.mesh_size)
        parameters = np.concatenate(
            [self.parameters[simulation_index, 3:], np.array([time_step_index], dtype=np.float32)]
        )
        return parameters, data


class NumpyDataset(DatDataset):
    def __getitem__(self, idx):
        simulation_index, time_step_index = self.get_indices(idx)
        data_file = osp.join(self.root, f"Res_{simulation_index}", f"data_{simulation_index}.npz")
        parameters = np.concatenate(
            [self.parameters[simulation_index, 3:], np.array([time_step_index], dtype=np.float32)]
        )
        with np.load(data_file, mmap_mode="r") as data:
            simulation_data = data["temperature"][time_step_index].copy()
        del data
        return parameters, simulation_data


def get_simulation_id(filename: str) -> int:
    return int(filename.split("_")[-1])


def get_time_step(filename: str) -> int:
    basename = osp.splitext(filename)[0]
    time_step = int(basename.split("_")[-1])
    return time_step


def get_parameters(result_folder):
    log_filename = osp.join(result_folder, "melissa_server_0.log")
    client_regex = re.compile(r"created client.\d+.sh with parameters")
    parameters_regex = re.compile(r"\[(?P<parameters>.+)\]")
    simulation_parameters = []
    with open(log_filename, "r") as log_file:
        for line in log_file:
            if re.search(client_regex, line):
                parameter_match = re.search(parameters_regex, line)
                assert parameter_match
                parameters_str = parameter_match.group("parameters")
                parameters = np.fromstring(parameters_str, sep=",", dtype=np.float32)
                simulation_parameters.append(parameters)

    parameters = np.array(simulation_parameters)
    del simulation_parameters
    return parameters


def process_data(result_folder: str, mesh_size: Tuple[int, int]):
    """For each simulation folder in the `result_folder`
    process the simulation data and retrieve the parameters to generate a numpy archive."""
    start = time.time()
    folders = sorted(glob.glob(f"{result_folder}/Res_*"), key=get_simulation_id)
    parameters = get_parameters(result_folder)
    for folder in folders:
        simulation_id = get_simulation_id(folder)
        print(f"Process folder {folder} for simulation {simulation_id}.")
        time_step_files = sorted(glob.glob(f"{folder}/**.dat"), key=get_time_step)
        simulation_data = np.stack(
            [np.loadtxt(f, usecols=1, dtype=np.float32) for f in time_step_files]
        )
        simulation_data = simulation_data.reshape(len(time_step_files), *mesh_size)
        np.savez(
            f"{folder}/data_{simulation_id}",
            temperature=simulation_data,
            parameters=parameters[simulation_id],
        )
    end = time.time()
    original_dataset_size = sum(
        [osp.getsize(file) for file in glob.glob(f"{result_folder}/**/*.dat")]
    )
    compressed_dataset_size = sum(
        [osp.getsize(file) for file in glob.glob(f"{result_folder}/**/*.npz")]
    )
    print(
        f"Dataset processing of {len(folders)} simulations"
        f" size {compressed_dataset_size / 1E6:.3f}MB ({original_dataset_size / 1E6:.3f}MB)"
        f" in {end - start:.3f}s"
    )


def create_dataset_folder(result_folder: str):
    """
    Create training and validation folders and sub-files
    """
    import os
    import shutil
    train = False
    out_folder = ""
    # create training and validation folders
    if "TRAINING" in result_folder:
        out_folder = "sc2023-heatpde-training"
        os.makedirs(os.path.join(out_folder), exist_ok=True)
        train = True
    elif "VALIDATION" in result_folder:
        out_folder = "sc2023-heatpde-validation"
        os.makedirs(os.path.join(out_folder), exist_ok=True)
    else:
        raise Exception("Wrong folder name")

    # move solutions to appropriate folders
    num_sol = len([folder for folder in os.listdir(result_folder) if "Res" in folder])
    data_ = []
    data: npt.NDArray = np.array([])
    parameters_ = []
    parameters: npt.NDArray = np.array([])
    for i in range(num_sol):
        # copy melissa log file
        if i == 0:
            shutil.copy(
                os.path.join(result_folder, "melissa_server_0.log"),
                f"{out_folder}/melissa_server_0.log",
            )
        # copy np binary results
        os.makedirs(os.path.join(out_folder, f"Res_{i}"), exist_ok=True)
        shutil.copy(
            os.path.join(result_folder, f"Res_{i}", f"data_{i}.npz"),
            f"{out_folder}/Res_{i}/data_{i}.npz",
        )
        # val solutions are loaded
        if not train:
            d = np.load(os.path.join(result_folder, f"Res_{i}/data_{i}.npz"))
            data_.append(d["temperature"])
            parameters_.append(d["parameters"])
    # val solutions are processed to extract npy files
    if not train:
        # network outputs are converted
        data = np.array(data_, dtype=np.float32)
        print(data.shape, data.dtype)
        # network inputs are reconstructed
        parameters = np.stack(parameters_)
        parameters = parameters[:, 3:]
        time_parameters_ = []
        time_parameters: npt.NDArray = np.array([])
        for p in parameters:
            for t in range(100):
                time_parameters_.append([*p, t])
        time_parameters = np.array(time_parameters_, dtype=np.float32)
        print(time_parameters)
        print(time_parameters.shape, time_parameters.dtype)
        # val inputs/outputs are saved
        np.save(os.path.join(out_folder, "validation_10.npy"), data)
        np.save(
            os.path.join(out_folder, "input_10.npy"), time_parameters
        )


def test_dataloading_time(
    result_folder: str,
    batch_size: int,
    num_workers: int,
    mesh_size: Tuple[int, int],
    dataset_class: Union[Type[DatDataset], Type[NumpyDataset]],
):

    from offline_tools import TensorboardLogger

    n_simulations = len(glob.glob(f"{result_folder}/Res_*"))
    n_time_steps = len(glob.glob(f"{result_folder}/**/*.dat")) // n_simulations
    dataset = dataset_class(result_folder, n_simulations, n_time_steps, mesh_size)
    batch_ctr = 0
    tb_logger = TensorboardLogger(0, f"Offline_{dataset_class.__name__}_{batch_size}_{num_workers}")
    for _ in torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    ):
        batch_ctr += 1
        tb_logger.log_scalar("batch", batch_ctr, batch_ctr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="Path to the output folders.")
    parser.add_argument("mesh_size", nargs=2, type=int, help="Size of the mesh.")
    parser.add_argument("--num_workers", type=int, help="Number of workers used.", default=2)
    parser.add_argument("--dataset_type", type=int, default=1, help="Dataset to use.")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--loading_time", action="store_true")
    parser.add_argument("--process", action="store_true")
    parser.add_argument("--create_folder", action="store_true")
    args = parser.parse_args()

    result_folder = osp.abspath(args.folder)
    mesh_size = args.mesh_size
    batch_size = args.batch_size
    num_workers = args.num_workers
    dataset_type = DatDataset if args.dataset_type == 1 else NumpyDataset
    if args.process:
        process_data(result_folder, mesh_size)
    if args.create_folder:
        create_dataset_folder(result_folder)
    if args.loading_time:
        test_dataloading_time(result_folder, batch_size, num_workers, mesh_size, dataset_type)
