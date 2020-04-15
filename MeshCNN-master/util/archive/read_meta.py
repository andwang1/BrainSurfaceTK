import numpy as np
import csv


path = '/vol/biomedic2/aa16914/shared/MScAI_brain_surface/data/meta_data.tsv'

def read_meta(path=path):
    '''Correctly reads a .tsv file into a numpy array'''
    data = []

    with open(path) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        data = []
        for idx, row in enumerate(rd):
            if idx == 0:
                continue
            data.append(row)

    data = np.array(data)

    return data

