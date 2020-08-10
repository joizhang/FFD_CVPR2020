import argparse

import matplotlib.pyplot as plt
import torch

import models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, help='directory for data')
    parser.add_argument('--arch', metavar='ARCH', default='vgg16', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1, help='manual seed')
    # parser.add_argument('--signature', default=str(datetime.datetime.now()))
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    # parser.add_argument('--save_dir', default='./runs', help='directory for result')
    opt = parser.parse_args()
    return opt


def train(args, model, train_loader, optimizer, cse_loss, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = cse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            template = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
            print(template.format(epoch, batch_idx * len(data), len(train_loader.dataset),
                                  100. * batch_idx / len(train_loader), loss.item()))


def test(model, test_loader, cse_loss):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += cse_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


def write_tfboard(writer, vals, itr, name):
    for idx, item in enumerate(vals):
        writer.add_scalar('data/%s%d' % (name, idx), item, itr)


def plot_image(data):
    plt.figure(figsize=(10, 5))
    for i in range(len(data)):
        image, label = data[i]
        print(i, image.size)
        plt.subplot(1, 5, i + 1)
        # plt.tight_layout()
        plt.title('Label {}'.format(label))
        plt.axis('off')
        plt.imshow(image)
        if i == 4:
            plt.show()
            break
