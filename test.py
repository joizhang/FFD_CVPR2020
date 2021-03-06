import argparse
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import accuracy_score, auc, roc_curve
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

import models
from datasets.classifier_dataset import DffdDataset
from tools.model_utils import AverageMeter, ProgressMeter, accuracy
from tools.train_utils import parse_args

torch.backends.cudnn.benchmark = True

PICKLE_FILE = "{}.pickle"


def show_metrics(args):
    with open(PICKLE_FILE.format(args.arch), "rb") as f:
        y_true, y_pred, y_score = pickle.load(f)
    print(len(y_true), len(y_pred), len(y_score))
    acc = accuracy_score(y_true, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, drop_intermediate=False)
    print(fpr, tpr)
    fnr = 1 - tpr
    eer = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
    tpr_0_01 = -1
    tpr_0_02 = -1
    tpr_0_05 = -1
    tpr_0_10 = -1
    tpr_0_20 = -1
    tpr_0_50 = -1
    tpr_1_00 = -1
    tpr_2_00 = -1
    tpr_5_00 = -1
    for i in range(len(fpr)):
        if fpr[i] > 0.0001 and tpr_0_01 == -1:
            tpr_0_01 = tpr[i - 1]
        if fpr[i] > 0.0002 and tpr_0_02 == -1:
            tpr_0_02 = tpr[i - 1]
        if fpr[i] > 0.0005 and tpr_0_05 == -1:
            tpr_0_05 = tpr[i - 1]
        if fpr[i] > 0.001 and tpr_0_10 == -1:
            tpr_0_10 = tpr[i - 1]
        if fpr[i] > 0.002 and tpr_0_20 == -1:
            tpr_0_20 = tpr[i - 1]
        if fpr[i] > 0.005 and tpr_0_50 == -1:
            tpr_0_50 = tpr[i - 1]
        if fpr[i] > 0.01 and tpr_1_00 == -1:
            tpr_1_00 = tpr[i - 1]
        if fpr[i] > 0.02 and tpr_2_00 == -1:
            tpr_2_00 = tpr[i - 1]
        if fpr[i] > 0.05 and tpr_5_00 == -1:
            tpr_5_00 = tpr[i - 1]
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold (AUC = %0.2f)' % roc_auc)
    metrics_template = "ACC: {:f} AUC: {:f} EER: {:f} TPR@0.01: {:f} TPR@0.10: {:f} TPR@1.00: {:f}"
    print(metrics_template.format(acc, roc_auc, eer, tpr_0_01, tpr_0_10, tpr_1_00))

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def test(test_loader, model, args):
    y_true = []
    y_pred = []
    y_score = []

    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(test_loader), [batch_time, top1], prefix='Test: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(test_loader):
            images, target = images.cuda(), target.cuda()
            y_true.extend(target.tolist())

            # compute output
            output = model(images)

            pred = torch.argmax(output, dim=1)
            y_pred.extend(pred.tolist())
            score, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
            y_score.extend(score.tolist())

            # measure accuracy and record loss
            acc1, = accuracy(output, target)
            top1.update(acc1[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
    pickle.dump([y_true, y_pred, y_score], open(PICKLE_FILE.format(args.arch), "wb"))
    return top1.avg


def main():
    args = parse_args()
    print(args)

    if args.resume:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        print("Loading checkpoint '{}'".format(args.resume))
        model = models.__dict__[args.arch](pretrained=False)
        model.cuda()
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])

        print("Initializing Data Loader")
        classes = {'Real': 0, 'Fake': 1}
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_data = DffdDataset(data_root=args.data_dir, mode='test', transform=transform, classes=classes,
                                seed=args.seed)
        test_loader = DataLoader(test_data, num_workers=1, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                 pin_memory=True)
        test(test_loader, model, args)
    else:
        show_metrics(args)


if __name__ == '__main__':
    # python test.py --data-dir /data/xinlin/mini-dffd --arch vgg16 --batch-size 100 --resume weights/vgg16_dffd.pt
    main()
