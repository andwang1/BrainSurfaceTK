import time
from options.train_options import TrainOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
from test import run_test

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = DataLoader(opt)
    dataset_size = len(dataset)
    print('#training meshes = %d' % dataset_size)

    model = create_model(opt)
    writer = Writer(opt)
    total_steps = 0

    best_val_cls_acc = -1
    best_val_reg_acc = 10000
    best_epoch = None

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + opt.epoch_count):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        train_loss_epoch = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            train_loss_epoch += model.loss

            if total_steps % opt.print_freq == 0:
                loss = model.loss
                t = (time.time() - iter_start_time) / opt.batch_size
                writer.print_current_losses(epoch, epoch_iter, loss, t, t_data)
                writer.plot_loss(loss, epoch, epoch_iter, dataset_size)

            # if i % opt.save_latest_freq == 0:
            #     print('saving the latest model (epoch %d, total_steps %d)' %
            #           (epoch, total_steps))
            #     model.save_network('latest')

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_network('latest')
            model.save_network(epoch)

        # Plot training loss per epoch on tensorboard
        writer.plot_epoch_loss(train_loss_epoch / len(dataset), epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay + opt.epoch_count, time.time() - epoch_start_time))
        if opt.verbose_plot:
            writer.plot_model_wts(model, epoch)

        if epoch % opt.run_test_freq == 0:
            acc = run_test(epoch)

            # Track the best model
            if opt.dataset_mode == 'regression' and acc < best_val_reg_acc:
                best_val_reg_acc = acc
                best_epoch = epoch
            elif opt.dataset_mode in ('classification', 'binary_class') and acc > best_val_cls_acc:
                best_val_cls_acc = acc
                best_epoch = epoch

            writer.plot_acc(acc, epoch)
        lr = model.update_learning_rate(acc, epoch)
        writer.plot_lr(lr, epoch)

    # At end of training, run the last model on the test set
    print("Final testing on model from epoch ", epoch)
    acc = run_test(epoch, is_val=False)
    writer.plot_test_acc(acc, epoch)

    # At end of training, pick best model and run a test on the test set
    print("Final testing on model from epoch ", best_epoch)
    acc = run_test(best_epoch, is_val=False)
    writer.plot_test_acc(acc, best_epoch)

    writer.close()
