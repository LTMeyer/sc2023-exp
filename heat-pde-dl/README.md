# SC2023 paper supporting materials

This document lists and describes all files used for the experiments presented in the paper as well as scripts for local purposes (see [our reproducibility guidelines](https://gitlab.inria.fr/melissa/sc2023/-/blob/main/README.md)).

## online experiments: `heat-pde-dl`

This is the main experiment folder which contains the scripts for online training with Melissa:

- `config_mpi.json`: Melissa configuration file for local online training
- `config_slurm.json`: Melissa configuration file used on the Jean-Zay supercomputer as a basis for the online experiments summarized in Table 2
- `config_reservoir_largescale.json`: Melissa configuration file used on Jean-Zay supercomputer for the online experiment introduced in Table 3

**Note**: the main differences between these two files lie in the resource configuration.

- `heatpde_dl_server_largescale.py`: Melissa server used on Jean-Zay supercomputer for the online experiment introduced in Tabled 3
- `heatpde_dl_server.py`: Melissa server used both locally and on Jean-Zay supercomputer for the online experiment introduced in Tabled 2

**Note**: the only difference between these two files lies in the learning rate milestones.

- `study_sg.sh`: bash script to be submitted on Jean-Zay supercomputer in order to perform online experiments

**Note**: the time and resources are to be adjusted to go from experiments of Table 2 to the experiment of Table 3

## offline experiments: `heat-pde-dl/offline`

This sub-folder contains all scripts for offline training, data generation and data conversion:

- `config_offline_mpi.json`: Melissa configuration file for local offline data generation
- `config_offline_slurm.json`: Melissa configuration file for offline data generation on Jean-Zay supercomputer
- `heatpde_offline_server.py`: Melissa simplified server to orchestrate the offline data generation both locally and on Jean-Zay supercomputer
- `offline_tools.py`: offline module containing functions used during the offline training
- `offline_training_js.sh`: bash script to be submitted on Jean-Zay supercomputer in order to perform the offline training

**Note**: this script is not to be used locally. 

- `process_dataset.py`: Python script used to convert `.dat` files to numpy binary files as well as to structure the validation and training datasets
- `process_dataset.sh`: bash script to be submitted on Jean-Zay supercomputer in order to execute `process_dataset.py`

**Note**: this script is not to be used locally. 

- `run_offline_study.py`: Python script performing the offline training by loading numpy binary files from disk
- `study_sg.sh`: bash script to be submitted on Jean-Zay supercomputer in order to perform the offline data generation

**Note**: this script is not to be used locally.

## Plotting scripts: `heat-pde-dl/plot`

- `plot_loss.py`: this script generates post-processing plots for the loss comparison
- `plot_throughput.py`: this script generates post-processing plots for the throughput comparison
- `convert_to_df.py`: this script converts the tensorboard logs from the experiments to a pandas dataframe for post processing.

