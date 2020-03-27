import pickle
from pyvista_examples import plot


if __name__ == '__main__':

    # for brain_idx in range(37):
    with open('./3/data_validation3.pkl', 'rb') as file:
        data, labels, pred = pickle.load(file)

        # print(len(data))
        plot(data, labels, pred)
#

