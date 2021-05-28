# coding=utf-8
"""
Outside wrapper to choose the training mode
"""

import argparse

from train_utils import train_source, train_target


def parseer():
    parser = argparse.ArgumentParser(description="Training mode selection and parameter definition")
    parser.add_argument("--training_mode", type=str, required=True,
                        help='training mode selection, only train_source, train_target, train_hybrid, dvrl_train'
                             'are supported')
    parser.add_argument("--model_dir", type=str, default='', help='train from last checkpoint. If empty,'
                                                                  'then train from scratch.')
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epoches', type=int, default=10, help='total training epoches')
    parser.add_argument('--save_epoch', type=int, default=2, help='save model frequency')
    parser.add_argument('--display_step', type=int, default=10, help='display training loss frequency')
    parser.add_argument('--val_freq', type=int, default=1, help='validation frequency')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parseer()
    if opt.training_mode == 'train_source':
        train_source.train(opt)

    elif opt.training_mode == 'train_target':
        train_target.train(opt)

