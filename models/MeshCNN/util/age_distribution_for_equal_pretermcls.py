import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Read the metadata file
meta = pd.read_csv("combined.tsv", delimiter='\t')

# Put participant_id and session_id together to get a unique key and use as index
meta['unique_key'] = meta['participant_id'] + "_" + meta['session_id'].astype(str)
meta.set_index('unique_key', inplace=True)
meta.drop(['participant_id', 'session_id', 'sedation', 'scan_number'], axis=1, inplace=True)


with open("indices.pk", "rb") as f:
    indices = pickle.load(f)

all_indices = indices['Train'] + indices['Val'] + indices['Test']

present_files_meta = meta.loc[all_indices]

preterm_indices = list(present_files_meta[present_files_meta['birth_age'] <= 38].index)
nonpreterm_indices = list(present_files_meta[present_files_meta['birth_age'] > 38].index)
nonpreterm_indices = nonpreterm_indices[::2] + nonpreterm_indices[::3]
sorted_indices = list(set(preterm_indices + nonpreterm_indices))
#
# #
test_indices = [index for index in sorted_indices[::6]]
val_indices = [index for index in sorted_indices[1::6]]
train_indices = list(set(sorted_indices).difference(set(test_indices).union(val_indices)))

# Check distributions are OK
f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)

sns.distplot(present_files_meta['birth_age'], ax=axes[0, 0], label='Full')
sns.distplot(present_files_meta.loc[train_indices, 'birth_age'], ax=axes[1, 0])
sns.distplot(present_files_meta.loc[test_indices, 'birth_age'], ax=axes[0, 1])
sns.distplot(present_files_meta.loc[val_indices, 'birth_age'], ax=axes[1, 1])
axes[0, 0].title.set_text('Full')
axes[0, 1].title.set_text('Test')
axes[1, 0].title.set_text('Train')
axes[1, 1].title.set_text('Val')
# plt.show()

# Have a look at certain ranges
specific_df = meta.loc[test_indices]
# print(specific_df.describe())
print(specific_df[specific_df['birth_age'] > 38].describe())


# Pickle this to reuse in scripts to split the data
indices = {"Train": train_indices, "Test": test_indices, "Val": val_indices}
with open("preterm_equal_split.pk", "wb") as f:
    pickle.dump(indices, f)