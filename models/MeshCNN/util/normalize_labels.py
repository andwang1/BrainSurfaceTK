def get_file_path(path, patient_id, session_id, extension):
    file_name = "sub-" + patient_id +"_ses-" + session_id + extension
    file_path = path + '/' + file_name
    return file_path


def get_all_unique_labels(meta_data):
    '''
    Return unique mapping of drawem features such that
        Original: [0, 3, 5, 7, 9 ...]
        Required: [0, 1, 2, 3, 4 ...]
        Mapping: [0:0, 3:1, 5:2, 7:3, 9:4, ...]
    :return: Mapping
    '''
    ys = []
    lens = []
    print('Getting unique labels by scanning all the files.')
    # # 3. Iterate through all patient ids
    for idx, patient_id in enumerate(tqdm(meta_data[:, 0])):            # Get file path to .vtk/.vtp for one patient
        file_path = get_file_path(path, patient_id, meta_data[idx, 1], extension)
        # If file exists
        if os.path.isfile(file_path):
            # print('Reading...')
            mesh = pv.read(file_path)
            y = torch.tensor(mesh.get_array('segmentation'))
            lens.append(y.size(0))
            ys.append(y)        # Now process the uniqueness of ys
    ys_concatenated = torch.cat(ys)
    # print(ys_concatenated)
    unique_labels = torch.unique(ys_concatenated)
    # print(unique_labels)
    unique_labels_normalised = unique_labels.unique(return_inverse=True)[1]
    self.num_labels = len(unique_labels)        # Create the mapping
    label_mapping = {}
    for original, normalised in zip(unique_labels, unique_labels_normalised):
        label_mapping[original.item()] = normalised.item()
    return label_mapping

def normalise_labels(self, y_tensor, label_mapping):
    '''
    Normalises labels in the format necessary for segmentation
    :return: tensor vector of normalised labels ([0, 3, 1, 2, 4, ...])
    '''
    # Having received y_tensor, use label_mapping
    temporary_list = []
    for y in y_tensor:
        temporary_list.append(label_mapping[y.item()])
    return torch.tensor(temporary_list)
