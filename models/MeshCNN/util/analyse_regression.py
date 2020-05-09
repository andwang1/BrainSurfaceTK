import imageio
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import seaborn as sns
import os
import re
sns.set()

# Plotting info
cutoff_age = 38
task = 'scan_age'
imgs_per_sec = 2
num_outliers = 5

# Load metadata
meta = pd.read_csv("combined.tsv", delimiter='\t')
# Put participant_id and session_id together to get a unique key and use as index
meta['unique_key'] = meta['participant_id'] + "_" + meta['session_id'].astype(str)
meta.set_index('unique_key', inplace=True)

# Find the min and max scan_age to fix plot sizes
results = pd.read_csv(f"test_logs/{task}/testacc_full_log_1.csv", header=0,
                      names=['unique_key', 'pred', 'label', 'error'])
min_actual_age = min(results.loc[:, 'label'])
max_actual_age = max(results.loc[:, 'label'])


# Find the number of epochs
files = os.listdir(f'test_logs/{task}')
re_pattern = r'[a-zA-Z_]+(\d+)\.csv$'
pattern = re.compile(re_pattern)
max_epoch = max(int(re.match(pattern, file).group(1)) for file in files)

def retrieve_plot_data(epoch):
    # Load test results
    results = pd.read_csv(f"test_logs/{task}/testacc_full_log_{epoch}.csv", header=0,
                          names=['unique_key', 'pred', 'label', 'error'])
    # Retrieve data for visualisation
    gender = [0 if indiv == 'Female' else 1 for indiv in meta.loc[results.loc[:, "unique_key"], 'gender']]
    pred = results.loc[:, "pred"]
    scan_age = results.loc[:, "label"]
    errors = results.loc[:, "error"].to_numpy()
    acc = np.round(np.mean(errors), 2)

    # Get the
    sorted_errors = np.argsort(np.abs(errors))
    outlier_indices = sorted_errors > sorted_errors.max() - num_outliers
    outlier_y_pred = pred[outlier_indices]
    outlier_x = scan_age[outlier_indices]
    outlier_lines = [(out_x, out_y_pred) for out_x, out_y_pred in zip(outlier_x, outlier_y_pred)]

    male = {"x": [], "y": []}
    female = {"x": [], "y": []}

    for label, actual_age, pred in zip(gender, scan_age, pred):
        if label == 0:
            female["x"].append(actual_age)
            female["y"].append(pred)
        else:
            male["x"].append(actual_age)
            male["y"].append(pred)
    return female, male, outlier_lines, acc

def generate_img(epoch):
    # Data for plotting
    print(epoch)
    female, male, outlier_lines, acc = retrieve_plot_data(epoch)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(**female, label="Female", color="red")
    ax.scatter(**male, label="Male", color="blue")
    ax.plot([min_actual_age, max_actual_age], [min_actual_age, max_actual_age], linestyle='dashed')
    ax.set(xlabel='Actual Scan Age', ylabel='Predicted', title=f'Scan Age Prediction - Epoch {epoch}')
    ax.set_ylim(min_actual_age - 1, max_actual_age + 1)

    for start, end in outlier_lines:
        ax.vlines(start, start, end, label=f"Top {num_outliers} Errors", linestyles="dashed")

    # To avoid printing multiple labels for vlines
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    textbox = f"L1: {acc}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(0.025, 0.75, textbox, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image

imageio.mimsave(f'./{task}.gif', [generate_img(i) for i in range(1, max_epoch)], fps=imgs_per_sec)
