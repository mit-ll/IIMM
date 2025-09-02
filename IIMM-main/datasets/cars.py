from torch.utils.data import DataLoader
from torchvision.datasets import StanfordCars

class Cars:
    def __init__(
            self,
            preprocess,
            location='./data',
            batch_size=32,
            num_workers=16
    ):
        # Data loading code

        self.train_dataset = StanfordCars(location, 'train', preprocess, download=False)
        self.train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = StanfordCars(location, 'test', preprocess, download=False)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        idx_to_class = dict((v, k)
                            for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace(
            '_', ' ') for i in range(len(idx_to_class))]