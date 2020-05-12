import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from joblib import Parallel, delayed
from selenium_testing import WebsiteTester

matplotlib.use('TkAgg')

website_url = "http://146.169.52.15:8000/"
login_user = "test"
login_pw = "test"

max_num_parallel = 24

def multiprocessing(num):
    print(f"Parallel Processes: {num}")
    tester = WebsiteTester(website_url, login_user, login_pw, verbose=False, headless=True)
    times = tester.headless_process(7201)
    return times


times = []
for i in range(1, max_num_parallel + 1):
    list_times = Parallel(n_jobs=-1)(delayed(multiprocessing)(url) for url in range(i))
    times.append(list_times)

predict_times = []
segment_times = []

for i, iteration in enumerate(times):
    assert i + 1 == len(iteration)
    total_time = np.zeros(2)
    for run in iteration:
        total_time += np.array(run)
    predict_times.append(total_time[0])
    segment_times.append(total_time[1])

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
    bar_access_times = plt.bar(bar_positions, predict_times, bar_width,
                               color='#ED0020',
                               label='Prediction Time')
    # Middle
    bar_predict_times = plt.bar(bar_positions, segment_times, bar_width - epsilon,
                                bottom=predict_times,
                                alpha=opacity,
                                color='blue',
                                edgecolor='blue',
                                linewidth=line_width,
                                label='Segmentation Time')

    plt.xticks(bar_positions, range(1, max_num_parallel + 1))
    plt.xlabel("Number of parallel executions")
    plt.ylabel('Time in seconds')
    plt.title("GUI Stress Testing")
    plt.legend(loc='best')
    sns.despine()
    # plt.show()
    plt.savefig("stress_test.pdf")