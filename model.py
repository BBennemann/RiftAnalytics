import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_features_entrada):
        super(Model, self).__init__()

        self.layer1 = nn.Linear(num_features_entrada, 64)
        
        self.layer2 = nn.Linear(64, 64)

        self.output = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x) 
        x = self.dropout(x)

        x = self.layer2(x)
        x = self.relu(x)

        x = self.output(x)
        x = self.sigmoid(x)

        return x