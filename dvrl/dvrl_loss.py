"""
Loss function for data valuation
"""

import torch
import torch.nn as nn


class DvrlLoss(nn.Module):
    def __init__(self, epsilon, threshold):
        """
        Construct class
        :param epsilon: Used to avoid log(0)
        :type epsilon:
        :param threshold: The exploration rate
        :type threshold:
        """
        super().__init__()
        self.epsilon = epsilon
        self.threshold = threshold

    def forward(self, est_data_value, s_input, reward_input):
        """
        Calculate the loss.
        :param est_data_value: The estimated data value(probability)
        :type est_data_value: torch.Tensor
        :param s_input: Final selection
        :type s_input: torch.Tensor
        :param reward_input: Reward
        :type reward_input: torch.Float
        :return:
        :rtype:
        """
        # Generator loss (REINFORCE algorithm)
        one = torch.ones_like(est_data_value, dtype=est_data_value.dtype)
        prob = torch.sum(s_input * torch.log(est_data_value + self.epsilon) + \
                         (one - s_input) * \
                         torch.log(one - est_data_value + self.epsilon))

        zero = torch.Tensor([0.0])
        zero = zero.to(est_data_value.device)

        dve_loss = (-reward_input * prob) + \
                   1e3 * torch.maximum(torch.mean(est_data_value) - self.threshold, zero) + \
                   1e3 * torch.maximum(1 - self.threshold - torch.mean(est_data_value), zero)

        return dve_loss
