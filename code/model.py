import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from .config import Config
from .mobilenetv2.mobilenetv2 import mobilenetv2

class BananasNet(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64):
        """
            Constructor that create BananasNet
            Params:
                - state_size: number of the state space.
                - action_size: number of the action space.
                - fc1_units: number of neurons of the first hidden layer.
                - fc2_units: number of neurons of the second hidden layer.
        """
        super(BananasNet, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """
            Forward that compute the mapping between 37 state -> 4 actions
            Params:
                - state: numpy array of (batch_size, 37)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PixelBananasNet(nn.Module):
    def __init__(self, state_size, action_size, alpha=1.0, pretrained=True):
        """
            Constructor that create PixelBananasNet
            Params:
                - state_size: number of the width state space.
                - action_size: number of the action space.
                - alpha: depth factor of mobilenet_v2.
                - pretrained: enable to load weights from imagenet.
        """
        super(PixelBananasNet, self).__init__()
        self.state_size_orig = state_size
        self.alpha = alpha
        self.state_size = math.ceil(state_size / 32 * alpha) * 32 * alpha
        self.net = mobilenetv2(input_size=96, width_mult=alpha)

        if pretrained:
            if alpha == 1.0:
                pretrained_dict = torch.load(os.path.join(Config.PRETRAINED_MOBILENET_V2_1))
            elif alpha == 0.5:
                pretrained_dict = torch.load(os.path.join(Config.PRETRAINED_MOBILENET_V2_05))
            elif alpha == 0.1:
                pretrained_dict = torch.load(os.path.join(Config.PRETRAINED_MOBILENET_V2_01))


            model_dict = self.net.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict) 
            self.net.load_state_dict(pretrained_dict)
        self.net.classifier = nn.Linear(self.net.classifier.in_features, action_size)

    def forward(self, state):
        """
            Forward that compute the mapping between 96x96 resized state -> 4 actions
            Params:
                - state: numpy array of (batch_size, 3, 96, 96)
        """
        return self.net(state)