"""Test script to classify target data."""

import torch
import torch.nn as nn

from utils import make_variable


def eval_tgt(encoder, classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0.
    acc = 0.

    # set loss function
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        # evaluate network
        for (images, labels) in data_loader:
            images = make_variable(images)
            # print(images.max(dim=1))
            labels = make_variable(labels).squeeze()

            preds = classifier(encoder(images))
            loss += criterion(preds, labels).item()
            # print(preds)

            pred_cls = preds.max(dim = 1)[1]
            acc += (pred_cls == labels).cpu().sum()
            # acc += pred_cls.eq(labels.data).cpu().sum()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
