from lenet import LeNet5
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import Dataset
import visdom
import onnx

def save_model(net):
    dummy_input = torch.randn(1, 1, 32, 32, requires_grad=True)
    torch.onnx.export(net, dummy_input, "lenet.onnx")

    onnx_model = onnx.load("lenet.onnx")
    onnx.checker.check_model(onnx_model)


def main():
    net = LeNet5()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=2e-3)

    dataset = Dataset()

    viz = visdom.Visdom()

    cur_batch_win = None

    for epoch, batch, loss in net.train_loss_gen(
        epochs=15,
        criterion=criterion,
        optimizer=optimizer,
        data_loader=dataset.train_loader):
        if batch == 0: # beginning of epoch
            loss_list, batch_list = [], []
        if batch == len(dataset.train_loader) - 1: # end of epoch
            total_correct = 0
            cum_loss = 0.0
            for _batch, loss, correct in net.test(criterion, dataset.test_loader):
                cum_loss += loss
                total_correct += correct
            avg_loss = cum_loss / dataset.test_len
            print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / dataset.test_len))
            save_model(net)
        loss_list.append(loss.detach().cpu().item())
        batch_list.append(batch + 1)

        if batch % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, batch, loss.detach().cpu().item()))

        # Update Visualization
        if viz.check_connection():
            cur_batch_win = viz.line(
                torch.Tensor(loss_list),
                torch.Tensor(batch_list),
                win=cur_batch_win,
                name='current_batch_loss',
                update=(None if cur_batch_win is None else 'replace'),
                opts={
                    'title': 'Epoch Loss Trace',
                    'xlabel': 'Batch Number',
                    'ylabel': 'Loss',
                    'width': 1200,
                    'height': 600})


if __name__ == '__main__':
    main()
