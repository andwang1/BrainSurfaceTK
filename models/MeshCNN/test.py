from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer

__author__ = "Rana Hanocka"
__license__ = "MIT"
__maintainer__ = "Andy Wang"

def run_test(epoch=-1, is_val=True):
    print('Running Test')
    opt = TestOptions().parse()
    # No shuffling for test set
    opt.serial_batches = True
    opt.which_epoch = epoch

    # Set batch_size to 1
    opt.batch_size = 1
    # If we are running on the test set change the folder path to where the test meshes are stored
    if not is_val:
        opt.phase = "test"

    dataset = DataLoader(opt)
    if opt.verbose:
        print("DEBUG testpath: ", opt.dataroot)
        print("DEBUG dataset length ", len(dataset))
    model = create_model(opt)
    writer = Writer(opt)
    writer.reset_counter()
    for i, data in enumerate(dataset):
        model.set_input(data)
        ncorrect, nexamples = model.test(epoch, is_val)
        if opt.verbose:
            print("DEBUG test ncorrect, nexamples ", ncorrect, nexamples)
        writer.update_counter(ncorrect, nexamples)
    writer.print_acc(epoch, writer.acc)
    return writer.acc


if __name__ == '__main__':
    opt = TestOptions().parse()
    run_test(opt.which_epoch, is_val=False)
