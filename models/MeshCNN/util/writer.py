import os
import time

__author__ = “Rana Hanocka”
__license__ = “MIT”
__maintainer__ = "Andy Wang"

try:
    from tensorboardX import SummaryWriter
except ImportError as error:
    print('tensorboard X not installed, visualizing wont be available')
    SummaryWriter = None


class Writer:
    def __init__(self, opt):
        self.name = opt.name
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.log_name = os.path.join(self.save_dir, 'loss_log.txt')
        self.testacc_log = os.path.join(self.save_dir, 'testacc_log.txt')
        self.start_logs()
        self.nexamples = 0
        self.ncorrect = 0
        #
        if opt.is_train and not opt.no_vis and SummaryWriter is not None:
            self.display = SummaryWriter(comment=opt.name)
        else:
            self.display = None

    def start_logs(self):
        """ creates test / train log files """
        if self.opt.is_train:
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)
        else:
            with open(self.testacc_log, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Testing Acc (%s) ================\n' % now)

    def print_current_losses(self, epoch, i, losses, t, t_data):
        """ prints train loss to terminal / file """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) loss: %.3f ' \
                  % (epoch, i, t, t_data, losses.item())
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def plot_loss(self, loss, epoch, i, n):
        iters = i + (epoch - 1) * n
        if self.display:
            self.display.add_scalar('data/train_loss', loss, iters)

    def plot_epoch_loss(self, loss, epoch):
        if self.display:
            self.display.add_scalar('data/train_loss_per_ep', loss, epoch)

    def plot_model_wts(self, model, epoch):
        if self.opt.is_train and self.display:
            for name, param in model.net.named_parameters():
                self.display.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

    def print_acc(self, epoch, acc):
        """ prints test accuracy to terminal / file """
        if self.opt.dataset_mode == "regression":
            message = 'epoch: {}, TEST ACC: [{:.5}]\n' \
                .format(epoch, acc)
        else:
            message = 'epoch: {}, TEST ACC: [{:.5} %]\n' \
                .format(epoch, acc * 100)
        print(message)
        with open(self.testacc_log, "a") as log_file:
            log_file.write('%s\n' % message)

    def plot_acc(self, acc, epoch):
        if self.display:
            self.display.add_scalar('data/val_acc', acc, epoch)

    def plot_test_acc(self, acc, epoch):
        if self.display:
            self.display.add_scalar('data/test_acc', acc, epoch)

    def plot_lr(self, lr, epoch):
        if self.display:
            self.display.add_scalar('data/lr', lr, epoch)

    def reset_counter(self):
        """
        counts # of correct examples
        """
        self.count = 0
        self.ncorrect = 0
        self.nexamples = 0

    def update_counter(self, ncorrect, nexamples):
        self.count += 1
        self.ncorrect += ncorrect
        self.nexamples += nexamples

    @property
    def acc(self):
        if self.opt.dataset_mode == "regression":
            return self.ncorrect / self.count
        return float(self.ncorrect) / self.nexamples

    def close(self):
        if self.display is not None:
            self.display.close()
