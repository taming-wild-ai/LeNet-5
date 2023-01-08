import unittest

class TestLeNetFast(unittest.TestCase):
    def test_ten_and_thirty_batches(self):
        from lenet import LeNet5
        import torch.nn as nn
        import torch.optim as optim
        from dataset import Dataset

        dataset = Dataset()
        net = LeNet5()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=2e-3)
        for epoch, batch, loss in net.train_loss_gen(
            epochs=15,
            criterion=criterion,
            optimizer=optimizer,
            data_loader=dataset.train_loader):
            if batch == 10:
                self.assertAlmostEqual(loss.item(), 2.0, 0)
            elif batch == 30:
                self.assertLess(loss.item(), 1.0)
                break

if __name__ == '__main__':
    unittest.main()
