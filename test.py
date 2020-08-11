import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import accuracy_score, auc
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

import models
from datasets.classifier_dataset import DffdDataset

torch.backends.cudnn.benchmark = True


def show_metrics():
    scores = []
    preds = []
    labels = []
    for iteration, batch in enumerate(loader):
        fname, image, label = batch[0], batch[1].cuda(), batch[2].cuda()
        print("Iteration: " + str(iteration))
        pred, score = test(image)
        scores.extend(score.select(1, 1).tolist())
        preds.extend(pred.tolist())
        labels.extend(label.tolist())
    print(scores)
    print(preds)
    print(labels)
    pickle.dump([scores, preds, labels], open("vgg_base.pickle", "wb"))
    acc = accuracy_score(labels, preds)
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, drop_intermediate=False)
    print(fpr)
    print(tpr)
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
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold (AUC = %0.2f)' % (roc_auc))
    print("ACC: {:f}\nAUC: {:f}\nEER: {:f}\nTPR@0.01: {:f}\nTPR@0.10: {:f}\nTPR@1.00: {:f}".format(acc, roc_auc, eer,
                                                                                                   tpr_0_01, tpr_0_10,
                                                                                                   tpr_1_00))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='root directory for data')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--save_dir', default='./runs', help='directory for result')
    parser.add_argument('--modeldir', help='model in pickle file to test')
    args = parser.parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print("Initializing Networks")
    model = models.__dict__[args.arch](pretrained=True)
    model.cuda()
    checkpoint = torch.load(args.modeldir)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    print("Initializing Data Loader")
    classes = {'Real': 0, 'Fake': 1}
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_data = DffdDataset(data_root=args.data_dir, mode='test', transform=transform, classes=classes, seed=args.seed)
    loader = DataLoader(test_data, num_workers=1, batch_size=args.batch_size, shuffle=False, drop_last=False,
                        pin_memory=True)

    # show_metrics()
