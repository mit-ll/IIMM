from torchvision.datasets import GTSRB as PyTorchGTSRB
from torch.utils.data import DataLoader

class GTSRB:
    def __init__(
            self,
            preprocess,
            location="./data",
            batch_size=128,
            num_workers=16
    ):

        # to fit with repo conventions for location
        self.train_dataset = PyTorchGTSRB(
            root=location,
            download=False,
            split='train',
            transform=preprocess
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        self.test_dataset = PyTorchGTSRB(
            root=location,
            download=False,
            split='test',
            transform=preprocess
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        # from https://github.com/openai/CLIP/blob/e184f608c5d5e58165682f7c332c3a8b4c1545f2/data/prompts.md
        self.classnames = [
            'red and white circle 20 kph speed limit',
            'red and white circle 30 kph speed limit',
            'red and white circle 50 kph speed limit',
            'red and white circle 60 kph speed limit',
            'red and white circle 70 kph speed limit',
            'red and white circle 80 kph speed limit',
            'end / de-restriction of 80 kph speed limit',
            'red and white circle 100 kph speed limit',
            'red and white circle 120 kph speed limit',
            'red and white circle red car and black car no passing',
            'red and white circle red truck and black car no passing',
            'red and white triangle road intersection warning',
            'white and yellow diamond priority road',
            'red and white upside down triangle yield right-of-way',
            'stop',
            'empty red and white circle',
            'red and white circle no truck entry',
            'red circle with white horizonal stripe no entry',
            'red and white triangle with exclamation mark warning',
            'red and white triangle with black left curve approaching warning',
            'red and white triangle with black right curve approaching warning',
            'red and white triangle with black double curve approaching warning',
            'red and white triangle rough / bumpy road warning',
            'red and white triangle car skidding / slipping warning',
            'red and white triangle with merging / narrow lanes warning',
            'red and white triangle with person digging / construction / road work warning',
            'red and white triangle with traffic light approaching warning',
            'red and white triangle with person walking warning',
            'red and white triangle with child and person walking warning',
            'red and white triangle with bicyle warning',
            'red and white triangle with snowflake / ice warning',
            'red and white triangle with deer warning',
            'white circle with gray strike bar no speed limit',
            'blue circle with white right turn arrow mandatory',
            'blue circle with white left turn arrow mandatory',
            'blue circle with white forward arrow mandatory',
            'blue circle with white forward or right turn arrow mandatory',
            'blue circle with white forward or left turn arrow mandatory',
            'blue circle with white keep right arrow mandatory',
            'blue circle with white keep left arrow mandatory',
            'blue circle with white arrows indicating a traffic circle',
            'white circle with gray strike bar indicating no passing for cars has ended',
            'white circle with gray strike bar indicating no passing for trucks has ended',
        ]
