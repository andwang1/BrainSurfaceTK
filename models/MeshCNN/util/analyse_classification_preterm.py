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

__author__ = "Andy Wang"
__license__ = "MIT"

# Plotting info
cutoff_age = 38
task = 'preterm'
x_var = 'birth_age'
x_label = x_var.split("_")
x_label = x_label[0].capitalize() + " " + x_label[1].capitalize()
imgs_per_sec = 2

# Load metadata
meta = pd.read_csv("combined.tsv", delimiter='\t')
# Put participant_id and session_id together to get a unique key and use as index
meta['unique_key'] = meta['participant_id'] + "_" + meta['session_id'].astype(str)
meta.set_index('unique_key', inplace=True)

# Find the number of epochs
files = os.listdir(f'test_logs/{task}')
re_pattern = r'[a-zA-Z_]+(\d+)\.csv$'
pattern = re.compile(re_pattern)
max_epoch = max(int(re.match(pattern, file).group(1)) for file in files)

def retrieve_plot_data(epoch):
    # Load test results
    results = pd.read_csv(f"test_logs/{task}/testacc_full_log_{epoch}.csv", header=0,
                          names=['unique_key', 'pred', 'label', 'prob'])
    # Retrieve data for visualisation
    actual_age = meta.loc[results.loc[:, "unique_key"], x_var].to_numpy()
    pred_prob = results.loc[:, "prob"]
    label = results.loc[:, "label"]
    acc = sum(results.loc[:, "pred"] == results.loc[:, "label"]) / len(results)
    acc = round(acc * 100)

    preterm = {"x": [], "y": []}
    notpreterm = {"x": [], "y": []}

    for label, actual_age, pred_prob in zip(label, actual_age, pred_prob):
        if label == 0:
            notpreterm["x"].append(actual_age)
            notpreterm["y"].append(pred_prob)
        else:
            preterm["x"].append(actual_age)
            preterm["y"].append(pred_prob)
    return notpreterm, preterm, acc

def generate_img(epoch):
    # Data for plotting
    print(epoch)
    notpreterm, preterm, acc = retrieve_plot_data(epoch)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(**notpreterm, label="Preterm", color="red")
    ax.scatter(**preterm, label="Not Preterm", color="blue")
    ax.vlines(cutoff_age, 0, 1, label="Preterm Cutoff", linestyles="dashed")
    ax.set(xlabel=f'{x_label}', ylabel='Predicted Probability (Preterm = 1)', title=f'Preterm Prediction - Epoch {epoch}')

    plt.legend(loc="upper left")

    textbox = f"Acc.: {acc}%"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(0.85, 0.95, textbox, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    ax.set_ylim(0, 1)

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image

imageio.mimsave(f'./{task}.gif', [generate_img(i) for i in range(1, max_epoch)], fps=imgs_per_sec)
