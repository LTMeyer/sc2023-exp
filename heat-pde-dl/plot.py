from typing import List

import pandas as pd
from tensorflow.python.summary.summary_iterator import summary_iterator
import matplotlib.pyplot as plt
from cycler import cycler


acm_colors = [
    "#6200d9",  # Purple
    "#0055c9",  # Dark Blue
    "#fc9200",  # Orange
    "#ff1924",  # Red
    "#00fafc",  # Blue
    "#00cf00",  # Green
    "#ffd600",  # Yellow
    "#82fcff",  # Light Blue
]
column_width_pts = 241.14749
pt_to_inch = 1 / 72.27
inch_to_cm = 2.54
colum_width = column_width_pts * pt_to_inch

plt.rc("axes", prop_cycle=cycler(color=acm_colors))
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Libertine",
        "figure.dpi": 300,
    }
)
w0, h0 = plt.rcParams["figure.figsize"]
plt.rcParams.update({"figure.figsize": [colum_width, h0 * colum_width / w0]})


def get_throughput(tb_file):
    events = summary_iterator(tb_file)
    size_data = []
    throughput_data = []
    start = 0
    for i, event in enumerate(events):
        if i == 0:
            start = event.wall_time
        if event.summary.value:
            if event.summary.value[0].tag == "samples_per_second":
                throughput_data.append(
                    {
                        "step": event.step,
                        "time": event.wall_time - start,
                        "throughput": event.summary.value[0].simple_value,
                    }
                )
            elif event.summary.value[0].tag == "buffer_size":
                size_data.append(
                    {
                        "step": event.step,
                        "time": event.wall_time - start,
                        "size": event.summary.value[0].simple_value,
                    }
                )
    return pd.DataFrame(size_data), pd.DataFrame(throughput_data)


def plot_reservoir_size(reservoir_df: List[pd.DataFrame], labels: List[str]):
    fig, ax = plt.subplots()
    for df, label in zip(reservoir_df, labels):
        ax.plot(df["time"], df["size"], label=label, lw=0.7)
    ax.set_xlabel(r"Time ($s$)")
    ax.set_ylabel(r"Reservoir population \#")
    fig.tight_layout()
    fig.savefig("reservoir_size.png", bbox_inches=None, pad_inches=0.1)
    plt.show()


def plot_reservoir_throughput(reservoir_df: List[pd.DataFrame], labels: List[str]):
    fig, ax = plt.subplots()
    for df, label in zip(reservoir_df, labels):
        ax.plot(df["time"], df["throughput"], label=label, lw=0.7)
    ax.set_xlabel(r"Time ($s$)")
    ax.set_ylabel(r"Throughput ($samples/s$)")
    fig.tight_layout()
    fig.savefig("reservoir_throughput.png")
    plt.show()


if __name__ == "__main__":
    tb_files = [
        "tensorboard/U1_SimpleQueue_1GPU_noval/gpu_0/"
        "events.out.tfevents.1680074930.r10i3n7.45215.0rank_0",
        "tensorboard/U1_ThresholdQueue_1GPU_noval/gpu_0/"
        "events.out.tfevents.1680080401.r10i1n2.687368.0rank_0",
        "tensorboard/U1_ThresholdEvictOnWriteQueue_1GPU_noval/gpu_0/"
        "events.out.tfevents.1680085107.r10i4n3.150083.0rank_0",
    ]
    labels = ["FIFO", "EvictionOnRead", "EvictionOnWrite"]
    reservoirs_df = [get_throughput(file) for file in tb_files]
    plot_reservoir_size([df[0] for df in reservoirs_df], labels)
    plot_reservoir_throughput([df[1] for df in reservoirs_df], labels)
