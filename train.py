import os
import random
import argparse
import time

import numpy as np
import torch
from torch import optim
from wilds import get_dataset as get_wilds_dataset
from wilds.common.data_loaders import get_train_loader as get_wilds_train_loader
from wilds.common.data_loaders import get_eval_loader as get_wilds_eval_loader
from wilds.common.grouper import CombinatorialGrouper

from utils import (
    create_model_and_transforms,
    get_tokenizer,
    get_tokenized_captions,
    pprint_args,
)
from train_helpers import train
from models.clip import CLIPAdapter, CustomCLIP, LoRACLIP, CLIPLinearProbe
from models.wrappers import OpenCLIP, SigLip, CoCa
from datasets.registry import get_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=['Cars','CIFAR100','DTD','EuroSAT','FMoW-id','FMoW-ood','GTSRB','ImageNet','MNIST','RESISC45','STL10','SUN397','SVHN'],
        required=True,
    )
    parser.add_argument(
        "--nepochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=128,
        help="Batch size",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=1e-4,
        help="Weight decay",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-6,
        help="Learning rate",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default='./ckpts',
        help='The directory in which models will be stored during training',
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Raise this flag to display a progress bar during training and evaluation",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["Adam", "SGD"],
        default="SGD",
        help="Optimization strategy to be used during training",
    )
    parser.add_argument(
        "--val-p",
        type=float,
        default=0.1,
        help='Percentage of the training data to use as validation data during training',
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default='clip',
        choices=['adapter', 'attention', 'bias', 'lora', 'probe', 'clip', 'coca', 'eva02', 'siglip'],
        help='Optional model variants to use. Default will use CLIP as provided by OpenCLIP',
    )
    parser.add_argument(
        "--train-blocks",
        nargs="?",
        type=int,
        help='If using the CLIP model variants "attention" or "bias", use this arg to specify the number of transformer blocks of which parameters are to be trained',
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="Rank for LoRA model",
    )
    parser.add_argument(
        "--reduction",
        type=int,
        default=4,
        help="Reduction to use for CLIP-Adapter"
    )
    parser.add_argument(
        "--lora-mlp",
        action="store_true",
        help="Optional flag when using LoRA model to apply LoRA to MLP weights, in addition to the attention weights",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="?",
        help="Option to manually set seed for training",
    )
    
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()
    pprint_args(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = create_model_and_transforms(args.model_name)
    tokenizer = get_tokenizer(args.model_name)
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    ### Data preparation ###
    if 'fmow' in args.dataset.lower():
        fmow = get_wilds_dataset(dataset="fmow", download=False, root_dir="./data")

        if 'ood' in args.dataset:
            subset = 'val'
            grouper = uniform_over_groups = None
        else:
            subset = 'train'
            uniform_over_groups = True
            grouper = CombinatorialGrouper(fmow, ["region"])
        
        data = fmow.get_subset(
            subset,
            transform=preprocess,
        )
        train_loader = get_wilds_train_loader(
            "standard",
            dataset=data,
            uniform_over_groups=uniform_over_groups,
            grouper=grouper,
            num_workers=args.num_workers,
            pin_memory=True,
            batch_size=args.bs,
        )
        data = fmow.get_subset(
            'id_val',
            transform=preprocess,
        )
        val_loader = get_wilds_eval_loader(
            "standard",
            dataset=data,
            num_workers=args.num_workers,
            pin_memory=True,
            batch_size=args.bs,
        )
    else:
        dataset_name = args.dataset+'Val'
        data = get_dataset(dataset_name=dataset_name,
                        preprocess=preprocess, 
                        batch_size=args.bs, 
                        num_workers=args.num_workers, 
                        val_fraction=args.val_p, 
                        max_val_samples=5000)
        
        train_loader = data.train_loader
        val_loader = data.test_loader

    # Create tokenized text captions
    tokenized_captions = get_tokenized_captions(args.dataset, data, tokenizer)


    ### Model Preparation ###
    print("Loading model..")
    if args.model_name == 'adapter':
        model = CLIPAdapter(clip_model=model, reduction=args.reduction)
    elif args.model_name == 'attention':
        model = CustomCLIP(clip_model=model)
        model.set_specified_train_layers(train_blocks=args.train_blocks, reg='attention')
        model = model.model
    elif args.model_name == 'bias':
        model = CustomCLIP(clip_model=model)
        model.set_specified_train_layers(train_blocks=args.train_blocks, reg='bias')
        model = model.model
    elif args.model_name == 'lora':
        model = LoRACLIP(clip_model=model, include_mlp=args.lora_mlp, rank=args.rank)
    elif args.model_name == 'probe':
        n_classes = len(data.classnames) if 'fmow' not in args.dataset.lower() else data.n_classes
        model = CLIPLinearProbe(clip_model=model, n_classes=n_classes)
    elif args.model_name == 'coca':
        model = CoCa(coca_model=model)
    elif args.model_name == 'siglip':
        model = SigLip(model=model)
    
    if args.model_name in ['attention', 'bias', 'clip', 'eva02']: model = OpenCLIP(model)
    model.to(device=device)

    ### Load optimizer ###
    print("Creating optimizer..")
    if args.optimizer == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.99),
            weight_decay=args.wd,
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.wd,
        )

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    model_save_path = os.path.join(args.model_dir, f"best_ckpt_{args.model_name}_{args.dataset}.pt")

    ### Train ###
    start_time = time.time()
    print("Beginning training..")
    model, best_val_acc = train(
        model,
        device=device,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        nepochs=args.nepochs,
        model_save_path=model_save_path,
        tokenized_text=tokenized_captions,
        progress=args.progress,
    )
    end_time = time.time()
    print("Training time elapsed:", end_time - start_time)

    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")
    
    