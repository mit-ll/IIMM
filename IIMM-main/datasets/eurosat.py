import pandas as pd
import torch
from torchvision.datasets import EuroSAT as PytorchEuroSAT
import re


def pretify_classname(classname):
    l = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', classname)
    l = [i.lower() for i in l]
    out = ' '.join(l)
    if out.endswith('al'):
        return out + ' area'
    return out

class EuroSAT:
    def __init__(
            self,
            preprocess,
            location="./data",
            batch_size=32,
            num_workers=16
        ):
        # Data loading code
        self.train_dataset = PytorchEuroSAT(
            root=location,
            download=False, 
            transform=preprocess,
        )
        
        self.test_dataset = PytorchEuroSAT(
            root=location,
            download=False, 
            transform=preprocess,
        )
        p = 0.9 # proportion data for train
        df = pd.DataFrame(self.train_dataset.samples, columns = ['paths', 'labels'])
        train_index = df.groupby('labels').sample(frac=p, random_state=1).index
            
        train_subset = list(df.loc[train_index][['paths','labels']].itertuples(index=False, name=None))
        test_subset = list(df[~df.index.isin(train_index)][['paths','labels']].itertuples(index=False, name=None))
        
        self.train_dataset.samples = train_subset
        self.test_dataset.samples = test_subset

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers
        )
        idx_to_class = dict((v, k)
                            for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace('_', ' ') for i in range(len(idx_to_class))]
        self.classnames = [pretify_classname(c) for c in self.classnames]
        ours_to_open_ai = {
            'annual crop': 'annual crop land',
            'forest': 'forest',
            'herbaceous vegetation': 'brushland or shrubland',
            'highway': 'highway or road',
            'industrial area': 'industrial buildings or commercial buildings',
            'pasture': 'pasture land',
            'permanent crop': 'permanent crop land',
            'residential area': 'residential buildings or homes or apartments',
            'river': 'river',
            'sea lake': 'lake or sea',
        }
        for i in range(len(self.classnames)):
            self.classnames[i] = ours_to_open_ai[self.classnames[i]]
