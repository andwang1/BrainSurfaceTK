import numpy as np
import pickle as pk
from joblib import Parallel, delayed
from selenium_testing import WebsiteTester

website_url = "http://146.169.53.170:8000/"
login_user = "slave"
login_pw = "slave"
session_id = 7201

max_num_parallel = 8

def multiprocessing(num):
    tester = WebsiteTester(website_url, login_user, login_pw, verbose=False, headless=False)
    times = tester.headless_process(session_id)
    return times


times = []
for i in range(1, max_num_parallel + 1):
    print(f"Parallel Processes: {i}")
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

with open(f"data.pk", "wb") as f:
    data = {"predict": predict_times, "segment": segment_times}
    pk.dump(data, f)

