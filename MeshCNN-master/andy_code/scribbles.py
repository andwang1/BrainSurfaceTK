import pickle

with open("indices.pk", "rb") as f:
    indices = pickle.load(f)

print(indices)