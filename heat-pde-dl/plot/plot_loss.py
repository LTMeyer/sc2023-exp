import os
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import copy
import matplotlib as mpl

column_width_pts = 241.14749
pt_to_inch = 1 / 72.27
inch_to_cm = 2.54
colum_width = column_width_pts * pt_to_inch

acm_colors = [
    "#ff1924",  # Red
    "#fc9200",  # Orange
    "#0055c9",  # Dark Blue
    "#6200d9",  # Purple
    "#00cf00",  # Green
    "#00fafc",  # Blue
    "#ffd600",  # Yellow
    "#82fcff",  # Light Blue
]

plt.rc("axes", prop_cycle=cycler(color=acm_colors))
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Libertine",
    "figure.dpi": 300,
    "font.size": 8,
    "axes.titlesize": "medium",
    "axes.labelsize": "medium"

})


w0, h0 = plt.rcParams["figure.figsize"]
plt.rcParams.update({"figure.figsize": [colum_width, h0 * colum_width / w0]})

root_folder = "../"
batch_folder = "tb_logs"
data_filename = "data.pkl"

value_names = ["val", "train"]
alpha = [1, 0.2]
lw = [1, 0.2]
lt = ["-", "--"]


# create a figure
fig, ax = plt.subplots()
df_full = pd.DataFrame()
n = 0
for subdir in os.listdir(os.path.join(root_folder, batch_folder)):
    data_path = os.path.join(
        root_folder, batch_folder, subdir, data_filename)
    print(f"gathering data from {data_path}")
    df = pd.read_pickle(data_path)
    folder_name = subdir

    df.name.replace("Loss/train", "train", inplace=True)
    df.name.replace("Loss/valid", "val", inplace=True)

    if "firo" in subdir:
        color = acm_colors[2]
    elif "fifo" in subdir:
        color = acm_colors[3]
    elif "reservoir" in subdir:
        color = acm_colors[1]
    elif "offline" in subdir:
        color = acm_colors[0]

    # for each value in the list of values
    for name, a, l, t in zip(value_names, alpha, lw, lt):
        print(f"one subdir {subdir} with {len(df)} values")
        print(f"asking for value {name} with alpha {a} and lw {l} and linestyle {t}")
        df_val = copy.deepcopy(df[df.name == name])
        print(f"df_val is {df_val} with shape {df_val.shape} and type")
        put = copy.deepcopy(df[df.name == "put_time"])
        print(f"put time {put['value']} with shape {put['value'].shape}")
        throughput = copy.deepcopy(df[df.name == "samples_per_second"])
        df_val = df_val.sort_values("step")

        if "offline" in subdir:
            start = df_val["wall_time"].values[0]
        else:
            start = put["wall_time"].values[0]
        stop = df_val["wall_time"].values[-1]
        print(f"plotting for subdir {subdir} with {len(df_val)} values")
        ax.plot(df_val["step"], df_val["value"].values, t,
                label=folder_name, alpha=a, lw=l, color=color)

        if "val" in name:
            print(f"Total time {(stop - start)/3600}")
            print(f"Minimum RMSE for {subdir} is {df_val['value'].min()})")
            print(f"Mean throughput {throughput['value'].mean()}\n")

    n += 1


# log-scale y-axis
plt.yscale("log")

# add limits to the y-axis
plt.ylim(1, 1e5)


# add labels and legend
plt.xlabel("# Batch")
plt.ylabel("Loss")

# create custom legend
labels = [
    "FIFO",
    "FIRO",
    "Reservoir",
    "Offline",
]
colors = [
    acm_colors[3],
    acm_colors[2],
    acm_colors[1],
    acm_colors[0],
]
legend_elements = [mpl.lines.Line2D([0], [0], color=c, label=tag) for c, tag in zip(colors, labels)]
legend_elements += [mpl.lines.Line2D([0], [0], color="k",
                                     linestyle="--", lw=.5, alpha=.3,
                                     label="Training"),
                    mpl.lines.Line2D([0], [0], color="k", label="Validation")]
_elemelegendnts = [mpl.lines.Line2D([0], [0], color=c, label=tag) for c, tag in zip(colors, labels)]
fig.legend(handles=legend_elements, ncols=2, loc="upper center", bbox_to_anchor=(
    0.7, 0.95), columnspacing=.5, handletextpad=.5, fontsize="small")

plt.tight_layout()
# convert value_names list to string
vs = "_".join(value_names)
# save the figure in ./figs using the batch_folder name and the value name
fig.savefig(f"./{batch_folder}_{vs.replace('/','_')}.png", bbox_inches="tight")

# show the figure
plt.show()
