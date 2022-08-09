import torch
import torch.nn as nn
import numpy as np

from hyperparameters import device
from architecture import ANN
from data import sc_X, X_train, y_train, X_test, y_test
from madgrad import MADGRAD

model = ANN(2).to(device)

criterion = nn.MSELoss()
optimizer = MADGRAD(model.parameters())

checkpoint = torch.load('model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

def get_pred(data):
  model.eval()

  with torch.no_grad():
    
    score = model(data)
    _, predictions = score.max(1)

    return predictions.numpy()

pred = get_pred(torch.from_numpy(X_test).to(device))

data_baru = np.array([[38, 80]], dtype=np.float32)
data_baru = sc_X.transform(data_baru)

pred = get_pred(torch.from_numpy(data_baru).to(device))

print(pred[0])

# num_correct = 0
# num_samples = 0

# score = model(torch.from_numpy(X_test).to(device))

# _, predictions = score.max(1)

# for i in range(len(predictions)):
#   if predictions[i] == y_test[i]:
#     num_correct += 1
    
# num_samples = predictions.size(0)

# acc = float(num_correct)/float(num_samples)*100

# print(predictions)