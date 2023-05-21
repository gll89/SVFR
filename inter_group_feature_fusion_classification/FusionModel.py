from torch import nn


class FusionModel(nn.Module):   # best_acc = 0.953947 epoch=6
    def __init__(self, pyramid):
        super(FusionModel, self).__init__()
        self.ft_shape_in = 1024*pyramid*3
        self.fc1 = nn.Linear(self.ft_shape_in, self.ft_shape_in)
        # self.fc2 = nn.Linear(self.ft_shape_in, self.ft_shape_in*2)
        # self.fc3 = nn.Linear(self.ft_shape_in*2, 2)
        self.fc2 = nn.Linear(self.ft_shape_in, self.ft_shape_in)
        self.fc3 = nn.Linear(self.ft_shape_in, 2)

    def forward(self, x):
        # import pdb;
        # pdb.set_trace()
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc2(x)
        # x = self.fc2(x)
        x = self.fc3(x)
        return x

