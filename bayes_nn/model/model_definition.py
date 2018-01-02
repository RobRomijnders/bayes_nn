import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from bayes_nn import conf

class Net(nn.Module):
    """
    Define a simple neural net
    """
    def __init__(self):
        super(Net, self).__init__()
        self.drop_prob = conf.drop_prob

        self.conv1 = nn.Conv2d(1, conf.num_filters, kernel_size=5)
        self.num_units = int((((28-5)+1)/2)**2*conf.num_filters)
        self.fc1 = nn.Linear(self.num_units, conf.num_fc)
        self.fc2 = nn.Linear(conf.num_fc, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = x.view(-1, self.num_units)
        x = F.dropout(x, training=self.training, p=self.drop_prob)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=self.drop_prob)
        x = self.fc2(x)
        return F.log_softmax(x)