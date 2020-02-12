import pickle
from pyvista_examples import plot


if __name__ == '__main__':

    with open('data.pkl', 'rb') as file:
        data, labels, pred = pickle.load(file)

    plot(data, labels, pred)