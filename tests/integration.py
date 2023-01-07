import unittest

class TestLeNet(unittest.TestCase):
    def test_fifteen_epochs(self):
        from lenet import LeNet5
        import torch.nn as nn
        import torch.optim as optim
        from torchvision.datasets.mnist import MNIST
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader

        data_train = MNIST('./data/mnist',
                        download=True,
                        transform=transforms.Compose([
                            transforms.Resize((32, 32)),
                            transforms.ToTensor()]))
        data_test = MNIST('./data/mnist',
                        train=False,
                        download=True,
                        transform=transforms.Compose([
                            transforms.Resize((32, 32)),
                            transforms.ToTensor()]))
        data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
        data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

        net = LeNet5()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=2e-3)
        for epoch in range(1, 16):
            net.train()
            for i, (images, labels) in enumerate(data_train_loader):
                optimizer.zero_grad()

                output = net(images)

                loss = criterion(output, labels)

                loss.backward()
                optimizer.step()
            net.eval()
        total_correct = 0
        avg_loss = 0.0
        for i, (images, labels) in enumerate(data_test_loader):
            output = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.detach().max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()

        avg_loss /= len(data_test)
        self.assertAlmostEqual(float(total_correct) / len(data_test), 0.99, 2)

if __name__ == '__main__':
    unittest.main()
