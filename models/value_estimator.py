"""
Implementation of Value Estimator(Selection Network)
"""

import torch
import torch.nn as nn


class ValueEstimator(nn.Module):
    """
    Selection Network
    """
    def __init__(self, hidden_num, layer_num, comb_dim):
        """
        Construct class
        :param hidden_num:
        :type hidden_num:
        :param layer_num:
        :type layer_num:
        :param comb_dim:
        :type comb_dim:
        """
        super().__init__()

        layers = []
        # input_dim+label_dim
        layers.append(nn.Linear(513, hidden_num))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(layer_num - 3):
            layers.append(nn.Linear(hidden_num, hidden_num))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(hidden_num, comb_dim))
        layers.append(nn.ReLU(inplace=True))

        self.net_1 = nn.Sequential(*layers)
        self.combine_layer = nn.Sequential(
            nn.Linear(comb_dim*2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y, y_hat_diff):
        """
        Feed forward.
        :param x: train_features
        :type x:
        :param y: train_labels
        :type y:
        :param y_hat_diff: l1 difference between predicion and grountruth
        :type y_hat_diff:
        :return:
        :rtype:
        """
        x = torch.cat([x, y], dim=1)
        x = self.net_1(x)
        x = torch.cat([x, y_hat_diff], dim=1)

        output = self.combine_layer(x)
        return output