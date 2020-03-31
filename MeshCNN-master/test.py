from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer


def run_test(epoch=-1):
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    # print("DEBUG epoch", epoch)
    # If testing outside of training, want the epoch number to be correct so the files are created correctly
    if epoch == -1:
        epoch = opt.which_epoch
    # print("DEBUG epoch", epoch)
    # Hardcode batchsize to 1
    opt.batch_size = 1
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()
    for i, data in enumerate(dataset):
        model.set_input(data)
        ncorrect, nexamples = model.test(epoch)
        writer.update_counter(ncorrect, nexamples)
    writer.print_acc(epoch, writer.acc)
    return writer.acc


if __name__ == '__main__':
    run_test()
