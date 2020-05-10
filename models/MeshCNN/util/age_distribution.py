import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

__author__ = "Andy Wang"
__license__ = "MIT"

# Read the metadata file
meta = pd.read_csv("combined.tsv", delimiter='\t')

# Put participant_id and session_id together to get a unique key and use as index
meta['unique_key'] = meta['participant_id'] + "_" + meta['session_id'].astype(str)
meta.set_index('unique_key', inplace=True)
meta.drop(['participant_id', 'session_id', 'sedation', 'scan_number'], axis=1, inplace=True)

# Get a list of all files that we currently have
os.chdir("datasets/all_brains")
obj_files = os.listdir(".")

# Extract the same unique keys from the filenames
unique_key_files = [obj_name[:-4] for obj_name in obj_files]

# Filter meta based on files we have
present_files_meta = meta.loc[unique_key_files]
# Some weird data issue prob from copying files, dropping noisy row
present_files_meta.drop(['present_obj_50'], inplace=True)

# General stats, who doesn't love numbers
print(present_files_meta.describe())

# Plot it out
sns.distplot(present_files_meta['scan_age'])
plt.show()

#### Redistribution of data for regression
sorted_indices = present_files_meta.sort_values('scan_age').index
# print(present_files_meta.sort_values('scan_age').head(20))
test_indices = [index for index in sorted_indices[::6]]
val_indices = [index for index in sorted_indices[1::6]]
train_indices = list(set(sorted_indices).difference(set(test_indices).union(val_indices)))

# Check distributions are OK
f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)

sns.distplot(present_files_meta['scan_age'], ax=axes[0, 0], label='Full')
sns.distplot(present_files_meta.loc[train_indices, 'scan_age'], ax=axes[1, 0])
sns.distplot(present_files_meta.loc[test_indices, 'scan_age'], ax=axes[0, 1])
sns.distplot(present_files_meta.loc[val_indices, 'scan_age'], ax=axes[1, 1])
axes[0, 0].title.set_text('Full')
axes[0, 1].title.set_text('Test')
axes[1, 0].title.set_text('Train')
axes[1, 1].title.set_text('Val')
plt.show()

# Have a look at certain ranges
train_val_comb_df = present_files_meta.loc[train_indices + test_indices]
print(train_val_comb_df[train_val_comb_df['scan_age'] > 40].describe())


# Pickle this to reuse in scripts to split the data
indices = {"Train": train_indices, "Test": test_indices, "Val": val_indices}
with open("indices_oldsplit.pk", "wb") as f:
    pickle.dump(indices, f)