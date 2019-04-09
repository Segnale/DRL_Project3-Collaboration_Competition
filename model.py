import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, device, state_size, action_size, fc1_units=256, fc2_units=128, seed=None):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.device = torch.device(device)
        if seed is not None:
            torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.to(self.device)

        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Actor(nn.Module):
    def __init__(self, device, state_size, action_size, fc1_units=256, fc2_units=256,  low=-1.0, high=1.0, seed=None):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed

        self.low = low
        self.high = high
        self.device = torch.device(device)
        if seed is not None:
            torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.to(self.device)

        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.clamp(torch.tanh(self.fc3(x)), self.low, self.high)
