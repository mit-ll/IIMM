import os

import torch
import torchvision.datasets as datasets


class DTD:
    def __init__(
            self,
            preprocess,
            location='./data',
            batch_size=32,
            num_workers=16,
            partition=1,
        ):
        # Data loading code
        # traindir = os.path.join(location, 'dtd', 'train')
        # valdir = os.path.join(location, 'dtd', 'val')

        # self.train_dataset = datasets.ImageFolder(
        #     traindir, transform=preprocess)
        

        # self.test_dataset = datasets.ImageFolder(valdir, transform=preprocess)

        self.train_dataset = datasets.DTD(
            root=location,
            split='train',
            partition=partition,
            download=False,
            transform=preprocess,
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = datasets.DTD(
            root=location,
            split='val',
            partition=partition,
            download=False,
            transform=preprocess,
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )

        self.classnames = self.train_dataset.classes

        # idx_to_class = dict((v, k)
        #                     for k, v in self.train_dataset.class_to_idx.items())
        # self.classnames = [idx_to_class[i].replace(
        #     '_', ' ') for i in range(len(idx_to_class))]