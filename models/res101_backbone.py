# coding=utf-8
"""
Dense121 Backbone for Covid image classfication
"""

import torch

from torch import nn
from torchvision.models import resnet101
from torchvision.models.resnet import ResNet, Bottleneck


class ResNet101Backbone(ResNet):
    """
    Model take Res101 as backbone. We only use the frontal part from pretrained model.
    """

    def __init__(self, pretrained=True):
        """
        Construct class
        :param pretrained: whether to load imagenet pretrained weights.
        :type pretrained: bool
        """
        super().__init__(Bottleneck, [3, 4, 23, 3])
        num_ftrs = 512 * 4

        # nn.CrossEntropyLoess() have softmax function inside, so here we don't apply softmax
        self.linear = nn.Linear(num_ftrs, 2)

        if pretrained:
            self.load_pretrained()

    def load_pretrained(self):
        """
        Load pretrained weights
        :return:
        :rtype:
        """
        pretrained_model = resnet101(pretrained=True)
        pretrained_dict = pretrained_model.state_dict()
        model_dict = self.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.load_state_dict(model_dict)

    def forward(self, x, finaly_layer=True):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if not finaly_layer:
            return x
        x = self.linear(x)
        return x


if __name__ == '__main__':
    data = torch.randn((2, 3, 256, 256))
    data = data.cuda()

    res_model = ResNet101Backbone()
    res_model.cuda()

    output = res_model(data)
    print(output)
