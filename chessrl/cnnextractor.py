import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class CustomCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomCNNExtractor, self).__init__(observation_space, features_dim)
        # Define a CNN for processing input of shape (8, 8, 12)
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate the output size of the CNN
        n_flatten = self.cnn(th.rand((12,8,8))).flatten().shape[0]
        print(n_flatten)
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        x = self.cnn(observations.permute(0, 3, 1, 2))  # Change to channels-first
        return self.linear(x)
