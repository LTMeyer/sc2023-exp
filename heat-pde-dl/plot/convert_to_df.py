import sys
import argparse

from tensorflow.python.summary.summary_iterator import summary_iterator
import pandas as pd
import os


def main() -> int:
    """
    Scans `--root-dir` for `gpu_0` folders, parses the tb events log within to
    create a dataframe. Saves the dataframe to the `gpu_0` folder as `data.pkl`
    """

    parser = argparse.ArgumentParser(
        description="Tensorboard to df converter"
    )

    parser.add_argument(
        "--root-dir",
        "-p",
        help="Absolute path to the general dir",
        type=str
    )

    args = parser.parse_args()
    root_dir = args.root_dir

    def convert_tfevent(filepath):
        return pd.DataFrame([
            parse_tfevent(e) for e in summary_iterator(filepath)
            if len(e.summary.value)
        ])

    def parse_tfevent(tfevent):
        return dict(
            wall_time=tfevent.wall_time,
            name=tfevent.summary.value[0].tag,
            step=tfevent.step,
            value=float(tfevent.summary.value[0].simple_value),
        )

    columns_order = ['wall_time', 'name', 'step', 'value']

    # Iterate through all directories in the root directory
    for directory in os.listdir(root_dir):
        print(f"Checking {directory}")
        if os.path.isdir(os.path.join(root_dir, directory)):
            # Check if there is a gpu_0 directory in the current directory
            log_dir = os.path.join(root_dir, directory)
            if os.path.exists(log_dir):
                for file in os.listdir(log_dir):
                    if "events.out.tfevents" not in file:
                        continue
                    event_file = os.path.join(log_dir, file)
                    if os.path.isfile(event_file):
                        out = []
                        print(f"Parsing {str(event_file)}")
                        out.append(convert_tfevent(str(event_file)))
                        df = pd.concat(out)[columns_order]
                        df.reset_index(drop=True)
                        df.to_pickle(log_dir + "/data.pkl")
                        break

    return 0


if __name__ == "__main__":
    sys.exit(main())
