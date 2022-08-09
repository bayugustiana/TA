import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ANN(nn.Module):
  def __init__(self, num_input):
    super(ANN, self).__init__()
    self.fc1 = nn.Linear(num_input, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 4)

  def forward(self, X):
    X = F.relu(self.fc1(X))
    X = F.relu(self.fc2(X))
    X = self.fc3(X)
    return X