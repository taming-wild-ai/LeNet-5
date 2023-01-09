import torch.nn as nn
from collections import OrderedDict


class C(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=6,
        conv_kernel_size=(5, 5),
        pool_kernel_size=(2, 2),
        pool_stride=2):
        super(C, self).__init__()

        self.c1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel_size)),
            ('relu1', nn.ReLU()),
            ('s1', nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride))
        ]))

    def forward(self, img):
        output = self.c1(img)
        return output


class C3(nn.Module):
    def __init__(self, in_channels=16, out_channels=120, kernel_size=(5, 5)):
        super(C3, self).__init__()

        self.c3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)),
            ('relu3', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.c3(img)
        return output


class F4(nn.Module):
    def __init__(self, in_features=120, out_features=84):
        super(F4, self).__init__()

        self.f4 = nn.Sequential(OrderedDict([
            ('f4', nn.Linear(in_features, out_features)),
            ('relu4', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.f4(img)
        return output


class F5(nn.Module):
    def __init__(self, in_features=84, out_features=10):
        super(F5, self).__init__()

        self.f5 = nn.Sequential(OrderedDict([
            ('f5', nn.Linear(in_features, out_features)),
            ('sig5', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.f5(img)
        return output


class LeNet5(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """
    def __init__(self):
        super(LeNet5, self).__init__()

        self.c1 = C()
        self.c2_1 = C(in_channels=6, out_channels=16)
        self.c2_2 = C(in_channels=6, out_channels=16)
        self.c3 = C3()
        self.f4 = F4()
        self.f5 = F5()

    def forward(self, img):
        output = self.c1(img)

        x = self.c2_1(output)
        output = self.c2_2(output)

        output += x

        output = self.c3(output)
        output = output.view(img.size(0), -1)
        output = self.f4(output)
        output = self.f5(output)
        return output


    def train_loss_gen(self, epochs, criterion, optimizer, data_loader):
        for epoch in range(1, epochs+1):
            self.train()
            for batch, (images, labels) in enumerate(data_loader):
                optimizer.zero_grad()

                output = self(images)

                loss = criterion(output, labels)

                yield epoch, batch, loss

                loss.backward()
                optimizer.step()


    def test(self, criterion, data_loader):
        self.eval()
        for batch, (images, labels) in enumerate(data_loader):
            output = self(images)
            loss = criterion(output, labels).sum()
            pred = output.detach().max(1)[1]
            correct = pred.eq(labels.view_as(pred)).sum()
            yield batch, loss, correct
