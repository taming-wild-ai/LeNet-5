import unittest

class TestLeNet(unittest.TestCase):
    def test_fifteen_epochs(self):
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
            pass
        total_correct = 0
        for _batch, loss, correct in net.test(criterion, dataset.test_loader):
            total_correct += correct
        self.assertAlmostEqual(float(total_correct) / dataset.test_len, 0.99, 2)

if __name__ == '__main__':
    unittest.main()
