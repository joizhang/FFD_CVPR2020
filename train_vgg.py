<<<<<<< HEAD
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch import hub
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.classifier_dataset import DATA
from models.vgg import vgg16
from tools.train_utils import parse_args, train, validate

torch.backends.cudnn.benchmark = True
hub.set_dir("/home/xinlin/data2/torch_home")


def main():
    args = parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # sig = str(datetime.datetime.now()) + args.signature
    # os.makedirs('%s/modules/%s' % (args.save_dir, sig), exist_ok=True)

    print("Initializing Data Loader")
    classes = {'Real': 0, 'Fake': 1}
    # img_paths = {'Real': [], 'Fake': []}
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_data = DATA(data_root=args.data_dir, mode='train', transform=transform, classes=classes, seed=args.seed)
    # plot_image(train_data)
    train_loader = DataLoader(train_data, num_workers=1, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              pin_memory=True)
    val_data = DATA(data_root=args.data_dir, mode='validation', transform=transform, classes=classes, seed=args.seed)
    val_loader = DataLoader(val_data, num_workers=1, batch_size=args.batch_size, shuffle=True, drop_last=True,
                            pin_memory=True)

    print("Initializing Networks")
    model = vgg16(pretrained=True)
    optimizer = optim.Adam(model.parameters())
    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    # writer = SummaryWriter('%s/logs/%s' % (args.save_dir, sig))

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    print("Start Training")
    for epoch in range(1, args.epochs + 1):
        train(train_loader, model, optimizer, criterion, epoch, args)
        validate(val_loader, model, criterion, args)

    torch.save(model.state_dict(), "dffd.pt")


if __name__ == '__main__':
    main()
=======
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch import hub
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from config import Config
from datasets.classifier_dataset import DATA
from models.vgg import vgg16
from tools.train_utils import parse_args, train, validate

torch.backends.cudnn.benchmark = True

CONFIG = Config()
hub.set_dir(CONFIG['TORCH_HOME'])


def main():
    args = parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # sig = str(datetime.datetime.now()) + args.signature
    # os.makedirs('%s/modules/%s' % (args.save_dir, sig), exist_ok=True)

    print("Initializing Data Loader")
    classes = {'Real': 0, 'Fake': 1}
    # img_paths = {'Real': [], 'Fake': []}
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_data = DATA(data_root=args.data_dir, mode='train', transform=transform, classes=classes, seed=args.seed)
    # plot_image(train_data)
    train_loader = DataLoader(train_data, num_workers=1, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              pin_memory=True)
    val_data = DATA(data_root=args.data_dir, mode='validation', transform=transform, classes=classes, seed=args.seed)
    val_loader = DataLoader(val_data, num_workers=1, batch_size=args.batch_size, shuffle=True, drop_last=True,
                            pin_memory=True)

    print("Initializing Networks")
    model = vgg16(pretrained=True)
    optimizer = optim.Adam(model.parameters())
    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    # writer = SummaryWriter('%s/logs/%s' % (args.save_dir, sig))

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    print("Start Training")
    for epoch in range(1, args.epochs + 1):
        train(train_loader, model, optimizer, criterion, epoch, args)
        validate(val_loader, model, criterion, args)

    torch.save(model.state_dict(), "dffd.pt")


if __name__ == '__main__':
    main()
>>>>>>> 7e743dd61a82fbc813ef1eea8ea792a43235814a
