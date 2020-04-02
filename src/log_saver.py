import os

def get_id(prefix=''):
    '''
    :return: The next expected id number of an experiment
             that hasn't yet been recorded!
    '''
    with open(f'{prefix}log_record.txt', 'r') as log_record:
        next_id = len(log_record.readlines()) + 1

    return str(next_id)

def save_to_log(experiment_description, prefix=''):
    '''
    Saves the experiment description into the log
    :param experiment_description: all the variables in a nice format
    '''

    add_first_line = False
    if not os.path.exists(f'{prefix}log_record.txt'):
        add_first_line = True

    if add_first_line:
        with open(f'{prefix}log_record.txt', 'a+') as log_record:
            log_record.write(f'LOG OF ALL THE EXPERIMENTS for {prefix}')

    with open(f'{prefix}log_record.txt', 'a+') as log_record:
        next_id = get_id(prefix=prefix)
        log_record.write('\n#{} ::: {}'.format(next_id, experiment_description))


if __name__ == '__main__':
    pass
     # # Model Parameters
    # lr = 0.001
    # batch_size = 8
    # num_workers = 2
    #
    # local_features = ['corr_thickness', 'myelin_map', 'curvature', 'sulc']
    # global_features = None
    # target_class = 'gender'
    # task = 'segmentation'
    # # number_of_points = 12000
    #
    # test_size = 0.09
    # val_size = 0.1
    # reprocess = False
    #
    # data = "reduced_50"
    # type_data = "inflated"
    #
    # log_descr = "LR=" + str(lr) + '\t\t'\
    #           + "Batch=" + str(batch_size) + '\t\t'\
    #           + "Num Workers=" + str(num_workers) + '\t'\
    #           + "Local features:" + str(local_features) + '\t'\
    #           + "Global features:" + str(global_features) + '\t'\
    #           + "Data used: " + data + '_' + type_data + '\t'\
    #           + "Split class: " + target_class
    #
    # save_to_log(log_descr)