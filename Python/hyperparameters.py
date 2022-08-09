import torch
import torch.nn as nn
from madgrad import MADGRAD
from architecture import ANN

device = torch.device('cpu')
model = ANN(2)
lr = 0.03
criterion = nn.CrossEntropyLoss()
optimizer = MADGRAD(model.parameters())