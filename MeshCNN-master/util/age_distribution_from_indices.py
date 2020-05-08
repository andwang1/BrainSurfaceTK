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

print(indices)

train_indices = indices['Train']
val_indices = indices['Val']
test_indices = indices['Test']

# Check distributions are OK
f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)

sns.distplot(meta['birth_age'], ax=axes[0, 0], label='Full')
sns.distplot(meta.loc[train_indices, 'birth_age'], ax=axes[1, 0])
sns.distplot(meta.loc[test_indices, 'birth_age'], ax=axes[0, 1])
sns.distplot(meta.loc[val_indices, 'birth_age'], ax=axes[1, 1])
axes[0, 0].title.set_text('Full')
axes[0, 1].title.set_text('Test')
axes[1, 0].title.set_text('Train')
axes[1, 1].title.set_text('Val')
plt.savefig('age_distribution.pdf')

plt.show()

# Have a look at certain ranges
specific_df = meta.loc[train_indices]
print(specific_df.describe())
print(specific_df[specific_df['birth_age'] > 38].describe())

