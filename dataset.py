from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class Dataset(object):
    def __init__(self):
        self.train_loader = DataLoader(
            MNIST(
                './data/mnist',
                download=True,
                transform=transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor()])),
            batch_size=256,
            shuffle=True,
            num_workers=8)

        data_test = MNIST('./data/mnist',
                        train=False,
                        download=True,
                        transform=transforms.Compose([
                            transforms.Resize((32, 32)),
                            transforms.ToTensor()]))
        self.test_len = len(data_test)
        self.test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)
