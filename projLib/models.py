import torch
import torch.nn.functional as F

class SimpleClassifier(torch.nn.Module):
    """
    Simple pytorch classifier which takes 20-dimensional inputs, has a fully
    connected hidden layer with 100 units and a single softmax output layer.
    """
    def __init__(self, input_size, hidden_size=100, output_size=2):
        super(SimpleClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)