import unittest
from lenet import LeNet5, C, C3, F4, F5

class TestLeNetFast(unittest.TestCase):
    def test_lenet5_f5_default(self):
        f5 = F5()
        self.assertEqual(f5.f5[0].in_features, 84)
        self.assertEqual(f5.f5[0].out_features, 10)

    def test_lenet5_f5_configured(self):
        config = {
            "in_features": 8,
            "out_features": 2 }
        f5 = F5(**config)
        self.assertEqual(f5.f5[0].in_features, 8)
        self.assertEqual(f5.f5[0].out_features, 2)

    def test_lenet5_f4_default(self):
        f4 = F4()
        self.assertEqual(f4.f4[0].in_features, 120)
        self.assertEqual(f4.f4[0].out_features, 84)

    def test_lenet5_f4_configured(self):
        config = {
            "in_features": 12,
            "out_features": 8 }
        f4 = F4(**config)
        self.assertEqual(f4.f4[0].in_features, 12)
        self.assertEqual(f4.f4[0].out_features, 8)

    def test_lenet5_c3_default(self):
        c3 = C3()
        self.assertEqual(c3.c3[0].in_channels, 16)
        self.assertEqual(c3.c3[0].out_channels, 120)
        self.assertEqual(c3.c3[0].kernel_size, (5, 5))

    def test_lenet5_c3_configured(self):
        config = {
            "in_channels": 1,
            "out_channels": 6,
            "kernel_size": (2, 2) }
        c3 = C3(**config)
        self.assertEqual(c3.c3[0].in_channels, 1)
        self.assertEqual(c3.c3[0].out_channels, 6)
        self.assertEqual(c3.c3[0].kernel_size, (2, 2))

    def test_lenet5_c_default(self):
        c1 = C()
        self.assertEqual(c1.c1[0].in_channels, 1)
        self.assertEqual(c1.c1[0].out_channels, 6)
        self.assertEqual(c1.c1[0].kernel_size, (5, 5))
        self.assertEqual(c1.c1[2].kernel_size, (2, 2))
        self.assertEqual(c1.c1[2].stride, 2)

    def test_lenet5_c_configured(self):
        c1_config = {
            "in_channels": 6,
            "out_channels": 16,
            "conv_kernel_size": (5, 5),
            "pool_kernel_size": (2, 2),
            "pool_stride": 2 }
        c1 = C(**c1_config)
        self.assertEqual(c1.c1[0].in_channels, 6)
        self.assertEqual(c1.c1[0].out_channels, 16)
        self.assertEqual(c1.c1[0].kernel_size, (5, 5))
        self.assertEqual(c1.c1[2].kernel_size, (2, 2))
        self.assertEqual(c1.c1[2].stride, 2)

    def test_ten_and_thirty_batches(self):
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
