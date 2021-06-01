# coding=utf-8
"""
Outside wrapper to choose the training mode
"""

import argparse

from train_utils import train_source, train_target, train_hybrid, train_dvrl


def parseer():
    parser = argparse.ArgumentParser(description="Training mode selection and parameter definition")
    parser.add_argument("--training_mode", type=str, required=True,
                        help='training mode selection, only train_source, train_target, train_hybrid, dvrl_train'
                             'are supported')
    parser.add_argument("--model_dir", type=str, default='', help='train from last checkpoint. If empty,'
                                                                  'then train from scratch.')
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--epoches', type=int, default=1, help='total training epoches')
    parser.add_argument('--save_epoch', type=int, default=2, help='save model frequency')
    parser.add_argument('--display_step', type=int, default=10, help='display training loss frequency')
    parser.add_argument('--val_freq', type=int, default=1, help='validation frequency')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')

    # dvrl related
    parser.add_argument('--inner_batch_size', type=int, default=256)
    parser.add_argument('--inner_learning_rate', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--comb_dim', type=int, default=1)
    parser.add_argument(
        '--layer_number',
        help='number of network layers',
        default=5,
        type=int)

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parseer()
    if opt.training_mode == 'train_source':
        train_source.train(opt)

    elif opt.training_mode == 'train_target':
        train_target.train(opt)

    elif opt.training_mode == 'train_hybrid':
        train_hybrid.train(opt)

    elif opt.training_mode == 'train_dvrl':
        train_dvrl.train(opt)