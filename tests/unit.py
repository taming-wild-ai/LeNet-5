import unittest

class TestLeNetFast(unittest.TestCase):
    def test_ten_and_thirty_batches(self):
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
        net.train()
        loss_list, batch_list = [], []
        for i, (images, labels) in enumerate(data_train_loader):
            optimizer.zero_grad()

            output = net(images)

            loss = criterion(output, labels)
            if i == 10:
                self.assertAlmostEqual(loss.item(), 2.0, 0)
            elif i == 30:
                self.assertLess(loss.item(), 1.0)
                break
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    unittest.main()
