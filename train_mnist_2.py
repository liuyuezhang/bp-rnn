from models.hebbian.fc import FC, SeriesModule
import argparse

import torch
import torch.nn.functional as F
import numpy as np

from torchvision import datasets, transforms


def train(args, model, train_loader, epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        # convert tensor to numpy
        data = data.numpy()
        target = F.one_hot(target, num_classes=10).float()
        target = target.numpy().reshape(10, 1)

        output = model.forward(data)
        error = output - target
        model.backward(error)

        loss = np.mean(error ** 2)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))


def test(args, model, test_loader):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # convert tensor to numpy
            data = data.numpy()
            target = F.one_hot(target, num_classes=10).float()
            target = target.numpy().reshape(10, 1)

            output = model.forward(data)
            error = output - target

            test_loss += np.mean(error ** 2)

            if output.argmax() == target.argmax():
                correct += 1

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True)

    lr = args.lr
    fc1 = FC(m=28 * 28, n=100, lr=lr)
    fc2 = FC(m=100, n=10, lr=lr)
    modules = [fc1, fc2]

    model = SeriesModule(modules)

    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, epoch)
        test(args, model, test_loader)


if __name__ == '__main__':
    main()
