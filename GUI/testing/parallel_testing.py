import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from joblib import Parallel, delayed
from selenium_testing import WebsiteTester

# Machine dependent
# matplotlib.use('Agg')
matplotlib.use('TkAgg')

website_url = "http://146.169.52.15:8000/"
login_user = "test"
login_pw = "test"
session_id = 7201

max_num_parallel = 4


def multiprocessing(num):
    tester = WebsiteTester(website_url, login_user, login_pw, verbose=False, headless=True)
    times = tester.full_upload_workflow(session_id, False)
    return times


times = []
for i in range(1, max_num_parallel + 1):
    print(f"Parallel Processes: {i}")
    list_times = Parallel(n_jobs=-1)(delayed(multiprocessing)(url) for url in range(i))
    times.append(list_times)

access_times = []
predict_times = []
segment_times = []

for i, iteration in enumerate(times):
    assert i + 1 == len(iteration)
    total_time = np.zeros(3)
    for run in iteration:
        total_time += np.array(run)
    access_times.append(total_time[0])
    predict_times.append(total_time[1])
    segment_times.append(total_time[2])

access_times = np.array(access_times)
predict_times = np.array(predict_times)
segment_times = np.array(segment_times)

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
    # Middle
    bar_predict_times = plt.bar(bar_positions, predict_times, bar_width - epsilon,
                                bottom=access_times,
                                alpha=opacity,
                                color='blue',
                                edgecolor='blue',
                                linewidth=line_width,
                                label='Prediction Time')
    # Top
    bar_segment_times = plt.bar(bar_positions, segment_times, bar_width - epsilon,
                                bottom=predict_times + access_times,
                                alpha=opacity,
                                color='green',
                                linewidth=line_width,
                                label='Segmentation Time')

    plt.xticks(bar_positions, range(1, max_num_parallel + 1))
    plt.xlabel("Number of parallel executions")
    plt.ylabel('Time in seconds')
    plt.title("GUI Stress Testing")
    plt.legend(loc='best')
    sns.despine()
    plt.show()
