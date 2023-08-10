import torch
import torch.nn as nn
from torch.autograd import Variable


class CNN(nn.Module):
    def __init__(self, num_filters, k_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=num_filters * 2, kernel_size=k_size, stride=1, padding=k_size // 2),

        )
        self.conv2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(num_filters, num_filters * 4, k_size, 1, k_size // 2),

        )
        self.conv3 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(num_filters * 2, num_filters * 8, k_size, 1, k_size // 2),

        )

        self.out = nn.AdaptiveAvgPool1d(1)
        self.layer1 = nn.Sequential(
            nn.Linear(1000, 1000 - 3 * (k_size - 1)),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(1000, 1000 - 3 * (k_size - 1)),
            nn.ReLU()
        )

    def reparametrize(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        # eps = torch.cuda.FloatTensor(std.size()).normal_(0,0.1)
        eps = torch.FloatTensor(std.size()).normal_(0, 0.1)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def forward(self, x):
        x = self.conv1(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        x = self.conv2(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        x = self.conv3(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        # print("feed:", x.shape)
        # output = self.out(x)
        # output = output.squeeze()
        output1 = self.layer1(x)
        output2 = self.layer2(x)
        output = self.reparametrize(output1, output2)
        # output = output.permute(0, 2, 1)
        return output, output1, output2



class decoder(nn.Module):
    def __init__(self, init_dim, num_filters, k_size, size):
        super(decoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(num_filters * 4, num_filters * 3),
            nn.ReLU()
        )

        self.convt = nn.Sequential(
            nn.ConvTranspose1d(num_filters * 3, num_filters * 2, k_size, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose1d(num_filters * 2, num_filters, k_size, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose1d(num_filters, 128, k_size, 1, 0),
            nn.ReLU(),
        )
        self.layer2 = nn.Linear(128, size)

    def forward(self, x, init_dim, num_filters, k_size):
        # print("Protein:", x.shape)
        x = self.layer(x)
        print("\nlayer 1:", x.size())
        # x = x.view(-1, num_filters * 3, init_dim - 3 * (k_size - 1))
        # print("\nlayer 1.5:", x.size())
        x = x.permute(0, 2, 1)
        x = self.convt(x)
        # print("\nlayer 2:", x.size())
        x = x.permute(0, 2, 1)
        x = self.layer2(x)
        print("final:",x.size())
        return x