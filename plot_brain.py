import pickle
from pyvista_examples import plot


if __name__ == '__main__':

    # for brain_idx in range(37):
    with open('./4/data_validation1.pkl', 'rb') as file:
        data, labels, pred = pickle.load(file)

        # print(len(data))
        plot(data, labels, pred)
#

