import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class GatedRecurrentConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 n_iter=3):
        super(GatedRecurrentConvLayer, self).__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_iter = n_iter

        self.conv_recurrent = nn.Conv2d(out_channels, out_channels,
                                        kernel_size=kernel_size, padding=padding, stride=stride)
        self.conv_recurrent_gated = nn.Conv2d(out_channels, out_channels,
                                              kernel_size=(1,1), padding=0, stride=(1,1))

        self.conv_forward = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                    kernel_size=kernel_size, padding=padding, stride=stride),
                                          nn.BatchNorm2d(out_channels),
                                          )
        self.conv_forward_gated = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                          kernel_size=(1,1), padding=0, stride=(1,1)),
                                                nn.BatchNorm2d(out_channels),
                                                )

        self.bn_recurrent = nn.ModuleList()
        self.bn_gate_recurrent = nn.ModuleList()
        self.bn_gate_mul = nn.ModuleList()

        for ii in range(n_iter):
            self.bn_recurrent.append(nn.BatchNorm2d(out_channels))
            self.bn_gate_recurrent.append(nn.BatchNorm2d(out_channels))
            self.bn_gate_mul.append(nn.BatchNorm2d(out_channels))

    def forward(self, x):
        # t = 0
        forward_conv = self.conv_forward(x)
        x_t = F.relu(forward_conv)
        gate_forward_conv = self.conv_forward_gated(x)

        # t > 0
        for timestep in range(self.n_iter):
            last_x = x_t

            # Calculate G(t)
            recurrent_gate = self.bn_gate_recurrent[timestep](self.conv_recurrent_gated(last_x))
            G_t = torch.sigmoid(gate_forward_conv + recurrent_gate)

            # Calculate x(t)
            recurrent = self.bn_recurrent[timestep](self.conv_recurrent(last_x))
            mul = self.bn_gate_mul[timestep](recurrent * G_t)
            x_t = F.relu(forward_conv + mul)

        return x_t


class GatedRecurrentConvNet(nn.Module):
    # This network was proposed in :
    # http://papers.nips.cc/paper/6637-gated-recurrent-convolution-neural-network-for-ocr.pdf

    def __init__(self, nclasses, in_channels=3, img_size=(64, 256)):
        super(GatedRecurrentConvNet, self).__init__()

        """
        Args :
            nclasses : The number of characters to output
            in_channels : The number of channels of input images
            img_size : The size of input images (Height, Width)
        """

        self.nclasses = nclasses
        self.in_channels = in_channels

        self.conv1 = nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels=64,
                                              kernel_size=(3,3), stride=1, padding=1),
                                    nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
                                    )
        self.GRCL1 = nn.Sequential(
                                    GatedRecurrentConvLayer(in_channels=64,
                                                             out_channels=64,
                                                             kernel_size=(3,3),
                                                             stride=(1,1),
                                                             padding=(1,1),
                                                             ),
                                    nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
                                    )
        self.GRCL2 = nn.Sequential(
                                    GatedRecurrentConvLayer(in_channels=64,
                                                             out_channels=128,
                                                             kernel_size=(3,3),
                                                             stride=(1,1),
                                                             padding=(1,1),
                                                             ),
                                    nn.MaxPool2d(kernel_size=(2,2),stride=(2,1), padding=(0,1))
                                    )
        self.GRCL3 = nn.Sequential(
                                    GatedRecurrentConvLayer(in_channels=128,
                                                             out_channels=256,
                                                             kernel_size=(3,3),
                                                             stride=(1,1),
                                                             padding=(1,1),
                                                             ),
                                    nn.MaxPool2d(kernel_size=(2,2),stride=(2,1), padding=(0,1))
                                    )

        if img_size[0] == 32:
            self.conv2 = nn.Conv2d(in_channels=256, out_channels=512,
                                    kernel_size=(2,2), stride=(1,1))
        elif img_size[0] == 64:
            self.conv2 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512,
                                                    kernel_size=(2,2), stride=(1,1)),
                                       nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
                                       )

        rnn_in = 512
        hidden = 256
        num_layers = 2
        self.lstm = nn.LSTM(input_size=rnn_in,
                            hidden_size=hidden,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True)

        self.fc = nn.Linear(hidden*2, nclasses)

    def forward(self, x):

        x = self.conv1(x)
        x = self.GRCL1(x)
        x = self.GRCL2(x)
        x = self.GRCL3(x)
        x = self.conv2(x)

        batchsize, channels, height, width = x.size()
        assert height == 1, "Height must be 1, but get {}".format(height)

        x = x.squeeze(2)
        x = x.permute(0, 2, 1)  # (batch_size, seq_length, features)
        x, _ = self.lstm(x)
        out = self.fc(x)

        return out
