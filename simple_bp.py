# write a simple back propagation linear regression model
# in pytorch
from torch import nn

import torch
import numpy as np

NUM_CATEGORY = 10
HIDDEN_SIZE = 32


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = None
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if self.linear is None:
            self.linear = nn.Linear(x.shape[1], NUM_CATEGORY)
        return self.softmax(self.linear(x))


# build fake data

x = torch.from_numpy(np.random.rand(100, HIDDEN_SIZE)).float()
y = torch.from_numpy(np.random.randint(0, NUM_CATEGORY, (100))).long()

# build model
model = Model()


# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# train
for epoch in range(100):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch} loss: {loss.item()}")
