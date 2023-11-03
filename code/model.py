import torch
import torch.nn as nn
from torchsummary import summary
from thop import profile, clever_format


def Conv1dBnReLU(in_channels, out_channels, stride, padding, group):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=5, stride=stride, padding=padding, groups=group),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True)
    )

class Model_groups(nn.Module):
    def __init__(self, classes=5):
        super(Model_groups, self).__init__()

        self.conv1_1 = Conv1dBnReLU(8, 4, 1, 0, 1)
        self.conv1_2 = Conv1dBnReLU(16, 8, 1, 0, 1)
        self.conv1_3 = Conv1dBnReLU(8, 4, 1, 0, 1)

        self.conv2 = Conv1dBnReLU(16, 32, 1, 0, 2)
        self.conv3 = Conv1dBnReLU(32, 16, 1, 0, 1)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)
        self.dropout = nn.Dropout(0.3)

        # ????
        self.fc1 = nn.Linear(16, classes)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """????"""
        g1 = x[:, 0:8, :]
        g2 = x[:, 8:24, :]
        g3 = x[:, 24:32, :]

        x1 = self.conv1_1(g1)
        x2 = self.conv1_2(g2)
        x3 = self.conv1_3(g3)

        x = torch.concat([x1, x2, x3], dim=1)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.maxpool(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)

        return x

if __name__ == "__main__":
    net = Model_groups()

    input_shape = (32, 128)
    print(summary(net, input_shape, device="cpu"))

    input_tensor = torch.randn(1, *input_shape)
    flops, params = profile(net, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print("FLOPs: %s" % (flops))
    print("params: %s" % (params))
