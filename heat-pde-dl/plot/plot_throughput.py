import os
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import copy

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

value_names = ["samples_per_second", "buffer_size"]
alpha = [1., 0.2]
lw = [1., 1.]
lt = ["-", "--"]

# create a figure with two vertical subplots
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
df_full = pd.DataFrame()
n = 0
for subdir in os.listdir(os.path.join(root_folder, batch_folder)):
    if "offline" in subdir:
        continue
    data_path = os.path.join(
        root_folder, batch_folder, subdir, data_filename)
    print(f"gathering data from {data_path}")
    df = pd.read_pickle(data_path)
    folder_name = subdir

    if "firo" in subdir:
        color = acm_colors[2]
        label = "FIRO"
    elif "fifo" in subdir:
        color = acm_colors[3]
        label = "FIFO"
    elif "reservoir" in subdir:
        color = acm_colors[1]
        label = "Reservoir"

    t = "-"
    a = 0.8

    df.name.replace("Loss/train", "train", inplace=True)
    df.name.replace("Loss/valid", "val", inplace=True)

    # for each value in the list of values
    for name, l in zip(value_names, lw):  # type: ignore
        if "buffer_size" in name:
            ax = ax2
        else:
            ax = ax1
        df_val = copy.deepcopy(df[df.name == name])
        df_val = df_val.sort_values("step")

        wall_time = df_val["wall_time"] - df_val["wall_time"].values[0]
        print(f"plotting for subdir {subdir} with {len(df_val)} values")
        # plot the data
        ax.plot(wall_time, df_val["value"].values, t,
                label=label, alpha=a, lw=l, color=color)
    n += 1


# add labels and legend
ax2.set_xlabel("Time (s)")
ax1.set_ylabel("Throughput\n(samples/s)", wrap=True, labelpad=8, ha="center")
handles, labels = fig.gca().get_legend_handles_labels()
print(handles)
order = [0, 1, 2]

ax2.set_ylabel("Population\n(samples)", wrap=True)

# add legend to plot
leg_list = [handles[idx] for idx in order]
lab_list = [labels[idx] for idx in order]
# plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
fig.legend(leg_list, lab_list, ncols=3, loc="upper center",
           fontsize="small", bbox_to_anchor=(0.6, 1.05))

plt.tight_layout()
# convert value_names list to string
vs = "_".join(value_names)
# save the figure in ./figs using the batch_folder name and the value name
fig.savefig(f"./{batch_folder}_{vs.replace('/','_')}.png", bbox_inches="tight")

# show the figure
plt.show()
