import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle as pk
import seaborn as sns

# Machine dependent
# matplotlib.use('Agg')
matplotlib.use('TkAgg')
# Fix plt cutting off the xlabel
plt.gcf().subplots_adjust(bottom=0.15)

with open("results/data_ray.pk", "rb") as f:
    data = pk.load(f)

access_times = data["access"]
predict_times = data["predict"]
segment_times = data["segment"]
max_num_parallel = len(segment_times)

with sns.axes_style("white"):
    sns.set_style()
    sns.set_style("ticks")
    sns.set_context("talk")

    # Plot details
    bar_width = 0.35
    epsilon = .009
    line_width = 1
    opacity = 0.7
    bar_positions = np.arange(1, max_num_parallel + 1)

    # Bottom section
    bar_access_times = plt.bar(bar_positions, access_times, bar_width,
                               color='#ED0020',
                               label='Access Time')
    # Middle section
    bar_predict_times = plt.bar(bar_positions, predict_times, bar_width - epsilon,
                                bottom=access_times,
                                alpha=opacity,
                                color='blue',
                                edgecolor='blue',
                                linewidth=line_width,
                                label='Prediction Time')
    # Top section
    bar_segment_times = plt.bar(bar_positions, segment_times, bar_width - epsilon,
                                bottom=predict_times + access_times,
                                alpha=opacity,
                                color='green',
                                linewidth=line_width,
                                label='Segmentation Time')

    plt.xticks(bar_positions, range(1, max_num_parallel + 1))
    plt.xlabel("Number of parallel executions")
    plt.ylabel('Average time in secs')
    plt.title("GUI Stress Testing")
    plt.legend(loc='best')
    sns.despine()
    plt.show()
