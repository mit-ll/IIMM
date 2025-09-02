from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import PIL.Image
import torch
from torchvision.datasets import VisionDataset

class SUN397:
    def __init__(
            self,
            preprocess,
            location="./data",
            batch_size=32,
            num_workers=16
    ):
        # Data loading code

        self.train_dataset = CustomSUN397(root=location, transform=preprocess, train=True)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = CustomSUN397(root=location, transform=preprocess, train=False)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        self.classnames = [v.replace('_', ' ').replace('/', ' ') for v in self.train_dataset.classes]


class CustomSUN397(VisionDataset):
    def __init__(
        self,
        root: str = './data',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        train: bool = True
    ) -> None:
        
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._data_dir = Path(self.root) / "SUN397"

        if not self._check_exists():
            raise RuntimeError("Dataset not found.")

        with open(self._data_dir / "ClassName.txt") as f:
            self.classes = [c[3:].strip() for c in f]
            
        if train:
            image_paths = "Training_01.txt"
        else:
            image_paths = "Testing_01.txt"

        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        with open(Path(self.root) / image_paths) as f:
            paths = f.read().splitlines()
        self._image_files = paths

        self._labels = [
            self.class_to_idx['/'.join(path.split('/')[2:-1])] for path in self._image_files
        ]
        
        
    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        path = self.root + "/SUN397/"
        image = PIL.Image.open(path + image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


    def _check_exists(self) -> bool:
        return self._data_dir.is_dir()
